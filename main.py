import asyncio
import os
from typing import Literal, Optional, TypedDict

from langchain_sarvam_ai import ChatSarvamAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

llm: Optional[ChatSarvamAI] = None
classifier = None


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def write_workflow_png(app, output_path: str) -> None:
    app.get_graph().draw_mermaid_png(output_file_path=output_path)


def reply_tool(email: dict, interrupt_value: object) -> str:
    if isinstance(interrupt_value, dict):
        suggested = interrupt_value.get("suggested_reply")
        if isinstance(suggested, str) and suggested:
            return suggested

    subject = email.get("subject", "your request")
    return (
        "Hi,\n\n"
        f"Thanks for the details about \"{subject}\". "
        "A teammate reviewed your message and will follow up with next steps shortly. "
        "If you have any additional details, please reply to this email.\n\n"
        "- Support Bot"
    )


async def print_checkpoint(app, config: dict, label: str) -> None:
    snapshot = await app.aget_state(config)
    step = snapshot.metadata.get("step") if snapshot.metadata else None
    checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
    interrupt_count = len(snapshot.interrupts)
    print(
        f"{label} checkpoint: step={step}, next={snapshot.next}, "
        f"interrupts={interrupt_count}, checkpoint_id={checkpoint_id}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FAQ KNOWLEDGE BASE
#   In a real system this would be a vector database. For the demo we keep
#   it as a plain dict so the audience can see exactly what's matched against.
# ─────────────────────────────────────────────────────────────────────────────

FAQ_DATABASE = {
    # ── Account & Login ────────────────────────────────────────────────────
    "password_reset": {
        "topic": "Resetting your password",
        "url":   "https://example.com/help/password-reset",
        "answer": "Click 'Forgot password' on the login page and follow the link sent to your inbox.",
    },
    "account_deletion": {
        "topic": "Deleting your account",
        "url":   "https://example.com/help/delete-account",
        "answer": "Go to Settings → Privacy → Delete Account. Your data is removed within 7 days.",
    },
    "change_email": {
        "topic": "Changing your email address",
        "url":   "https://example.com/help/change-email",
        "answer": "Settings → Account → Email. We'll send a verification link to your new address.",
    },
    "two_factor_auth": {
        "topic": "Setting up two-factor authentication",
        "url":   "https://example.com/help/2fa",
        "answer": "Settings → Security → Two-Factor Authentication. We support authenticator apps and SMS.",
    },
    "username_change": {
        "topic": "Changing your username",
        "url":   "https://example.com/help/username",
        "answer": "Settings → Profile → Username. Note: usernames can only be changed once every 30 days.",
    },

    # ── Orders & Shipping ──────────────────────────────────────────────────
    "shipping_time": {
        "topic": "Shipping and delivery times",
        "url":   "https://example.com/help/shipping",
        "answer": "Standard shipping takes 3-5 business days. Express is 1-2 business days.",
    },
    "order_tracking": {
        "topic": "Tracking your order",
        "url":   "https://example.com/help/track-order",
        "answer": "You'll receive a tracking link by email once your order ships. You can also view it under My Orders.",
    },
    "international_shipping": {
        "topic": "International shipping",
        "url":   "https://example.com/help/international",
        "answer": "We ship to 60+ countries. Customs fees and import duties are the buyer's responsibility.",
    },
    "change_shipping_address": {
        "topic": "Changing your shipping address",
        "url":   "https://example.com/help/change-address",
        "answer": "Address can be edited up to 1 hour after ordering. After that, contact support before the order ships.",
    },

    # ── Payments & Refunds ─────────────────────────────────────────────────
    "refund_policy": {
        "topic": "Refund policy",
        "url":   "https://example.com/help/refunds",
        "answer": "We offer full refunds within 30 days of purchase. No questions asked.",
    },
    "payment_methods": {
        "topic": "Accepted payment methods",
        "url":   "https://example.com/help/payments",
        "answer": "We accept Visa, Mastercard, Amex, PayPal, Apple Pay, Google Pay, and UPI in select regions.",
    },
    "discount_codes": {
        "topic": "Applying discount codes",
        "url":   "https://example.com/help/discounts",
        "answer": "Enter your code at checkout in the 'Promo code' field. Only one code can be used per order.",
    },
    "invoice_request": {
        "topic": "Getting an invoice or receipt",
        "url":   "https://example.com/help/invoices",
        "answer": "Invoices are auto-emailed after purchase. You can also download them from My Orders → Invoice.",
    },

    # ── Returns & Warranty ─────────────────────────────────────────────────
    "return_process": {
        "topic": "How to return an item",
        "url":   "https://example.com/help/returns",
        "answer": "Start a return from My Orders → Return. Print the prepaid label and drop it at any carrier location.",
    },
    "warranty_info": {
        "topic": "Product warranty",
        "url":   "https://example.com/help/warranty",
        "answer": "All products carry a 1-year manufacturer warranty. Premium items have a 2-year warranty.",
    },

    # ── Subscription & Billing ─────────────────────────────────────────────
    "subscription_cancel": {
        "topic": "Cancelling your subscription",
        "url":   "https://example.com/help/cancel-subscription",
        "answer": "Settings → Subscription → Cancel. Your access remains active until the end of the current billing cycle.",
    },
    "free_trial": {
        "topic": "Free trial details",
        "url":   "https://example.com/help/free-trial",
        "answer": "We offer a 14-day free trial. No credit card required to start. Cancel anytime before day 14 to avoid charges.",
    },
    "upgrade_plan": {
        "topic": "Upgrading or downgrading your plan",
        "url":   "https://example.com/help/change-plan",
        "answer": "Settings → Subscription → Change Plan. Upgrades apply immediately; downgrades take effect next cycle.",
    },

    # ── Product & Support ──────────────────────────────────────────────────
    "mobile_app": {
        "topic": "Downloading the mobile app",
        "url":   "https://example.com/help/mobile-app",
        "answer": "Available on iOS App Store and Google Play. Search 'YourCompany' and look for our official logo.",
    },
    "support_hours": {
        "topic": "Customer support hours",
        "url":   "https://example.com/help/contact",
        "answer": "Our team is available Monday-Friday, 9am-6pm IST. Average response time is under 4 hours on weekdays.",
    },
    "gift_cards": {
        "topic": "Buying or redeeming gift cards",
        "url":   "https://example.com/help/gift-cards",
        "answer": "Gift cards can be purchased from the Shop → Gift Cards page. Redeem at checkout using the unique code.",
    },
    "data_export": {
        "topic": "Exporting your data",
        "url":   "https://example.com/help/data-export",
        "answer": "Settings → Privacy → Export Data. We'll email you a download link within 24 hours.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class EmailState(TypedDict):
    # Incoming email
    sender: str
    recipient: str
    date: str
    subject: str
    content: str

    # Filled by the classifier node
    route: Literal["FAQ", "Human", "Invalid"]
    faq_key: Optional[str]           # key into FAQ_DATABASE if route == "FAQ"
    reasoning: str

    # Filled by whichever responder node runs
    auto_reply: str


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER NODE
# ─────────────────────────────────────────────────────────────────────────────

class Classification(BaseModel):
    route: Literal["FAQ", "Human", "Invalid"] = Field(
        description="Pick one: 'FAQ', 'Human', or 'Invalid'",
    )
    faq_key: Optional[str] = Field(
        default=None,
        description="If route is 'FAQ', the matching key from the FAQ database. Otherwise None.",
    )
    reasoning: str = Field(description="One sentence explaining the decision")


def init_llm_and_classifier() -> None:
    global llm, classifier
    if llm is None:
        # Structured output is supported on sarvam-30b and sarvam-105b.
        llm = ChatSarvamAI(model="sarvam-30b")
        classifier = llm.with_structured_output(Classification)


def classifier_node(state: EmailState) -> dict:
    """Reads the email and decides: FAQ auto-reply, human review, or invalid."""
    init_llm_and_classifier()
    if classifier is None:
        raise RuntimeError("LLM not initialized")

    faq_summary = "\n".join(
        f"  - {key}: {entry['topic']}" for key, entry in FAQ_DATABASE.items()
    )

    result: Classification = classifier.invoke([
        SystemMessage(content=(
            "You are an email triage assistant. Read the incoming email and decide:\n"
            "  - Set route='FAQ' ONLY if the email's question clearly matches one "
            "of the FAQ topics below. Then set faq_key to that exact key.\n"
            "  - Set route='Human' for anything ambiguous, complex, complaint-related, "
            "or not in the FAQ. Then leave faq_key as null.\n"
            "  - Set route='Invalid' if the email is unrelated to customer support. "
            "Then leave faq_key as null.\n\n"
            f"Available FAQ entries:\n{faq_summary}"
        )),
        HumanMessage(content=(
            f"From: {state['sender']}\n"
            f"To: {state['recipient']}\n"
            f"Date: {state['date']}\n"
            f"Subject: {state['subject']}\n\n"
            f"{state['content']}"
        )),
    ])

    print(f"\nClassifier decision: {result.route}")
    print(f"Reasoning: {result.reasoning}")
    if result.faq_key:
        print(f"Matched FAQ: {result.faq_key}")

    return {
        "route":     result.route,
        "faq_key":   result.faq_key,
        "reasoning": result.reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE — the heart of the routing
# ─────────────────────────────────────────────────────────────────────────────

def route_decision(state: EmailState) -> str:
    """Returns the name of the next node based on the classifier's choice."""
    if state["route"] == "FAQ" and state["faq_key"] in FAQ_DATABASE:
        return "faq_responder"
    if state["route"] == "Invalid":
        return "invalid_request"
    return "human_review"


# ─────────────────────────────────────────────────────────────────────────────
# FAQ RESPONDER NODE
# ─────────────────────────────────────────────────────────────────────────────

def faq_responder_node(state: EmailState) -> dict:
    """Generates an auto-reply pointing the user to the matching FAQ page."""
    entry = FAQ_DATABASE[state["faq_key"]]

    reply = (
        f"Hi,\n\n"
        f"Thanks for reaching out. It looks like your question is about "
        f"\"{entry['topic']}\".\n\n"
        f"{entry['answer']}\n\n"
        f"You can find more detail here: {entry['url']}\n\n"
        f"If this doesn't solve your issue, just reply to this email and we'll "
        f"loop in a teammate.\n\n"
        f"- Support Bot"
    )
    print("\nFAQ auto-reply generated.")
    return {"auto_reply": reply}


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN REVIEW NODE  (simulated)
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: EmailState) -> dict:
    """
    Simulates forwarding to a human teammate.
    Sends back an automated 'we got your message' reply to the user, and
    asks the LLM to draft a fake internal Slack/Teams ping for the demo.
    """
    init_llm_and_classifier()
    if llm is None:
        raise RuntimeError("LLM not initialized")
    suggested_reply = (
        f"Hi,\n\n"
        f"Thanks for your email regarding \"{state['subject']}\". "
        f"We've forwarded this to our team for review and you'll hear back "
        f"within one business day.\n\n"
        f"- Support Bot"
    )

    human_reply = interrupt({
        "type": "human_reply",
        "sender": state["sender"],
        "subject": state["subject"],
        "suggested_reply": suggested_reply,
    })

    # Fake internal handoff message generated by the LLM for the demo
    handoff_msg = llm.invoke([
        SystemMessage(content=(
            "Generate a one-line internal Slack ping to the support team about "
            "an incoming customer email that needs human review. Keep it under "
            "30 words. Include the sender and a 5-word summary of the issue."
        )),
        HumanMessage(content=(
            f"Sender: {state['sender']}\n"
            f"Subject: {state['subject']}\n"
            f"Body: {state['content']}"
        )),
    ]).content

    reply_text = human_reply if isinstance(human_reply, str) and human_reply else suggested_reply

    print("\nHuman review path triggered.")
    print(f"Internal ping: {handoff_msg}")
    print("Auto-reply sent to user.")

    return {"auto_reply": reply_text}


# ─────────────────────────────────────────────────────────────────────────────
# INVALID REQUEST NODE
# ─────────────────────────────────────────────────────────────────────────────

def invalid_request_node(state: EmailState) -> dict:
    auto_reply = (
        "Hi,\n\n"
        "Thanks for reaching out. This inbox is for customer support questions only. "
        "If you have a support issue, please reply with details so we can help.\n\n"
        "- Support Bot"
    )
    print("\nInvalid request path triggered.")
    return {"auto_reply": auto_reply}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(EmailState)

    graph.add_node("classifier",     classifier_node)
    graph.add_node("faq_responder",  faq_responder_node)
    graph.add_node("human_review",   human_review_node)
    graph.add_node("invalid_request", invalid_request_node)

    graph.add_edge(START, "classifier")

    # Conditional edge — the classifier's decision routes to one of two nodes
    graph.add_conditional_edges(
        "classifier",
        route_decision,
        {
            "faq_responder": "faq_responder",
            "human_review":  "human_review",
            "invalid_request": "invalid_request",
        },
    )

    graph.add_edge("faq_responder", END)
    graph.add_edge("human_review",  END)
    graph.add_edge("invalid_request", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_EMAILS = [
    {
        "sender":    "alice@example.com",
        "recipient": "support@yourcompany.com",
        "date":      "2025-05-06",
        "subject":   "Can't log in — forgot my password",
        "content":   "Hi, I tried to log into my account but I can't remember my password. How do I reset it?",
    },
    {
        "sender":    "bob@example.com",
        "recipient": "support@yourcompany.com",
        "date":      "2025-05-06",
        "subject":   "URGENT: charged twice for the same order",
        "content":   "I was charged twice for order #4421 yesterday. This is the third time this has happened. I want to speak with a manager today.",
    },
    {
        "sender":    "carol@example.com",
        "recipient": "support@yourcompany.com",
        "date":      "2025-05-06",
        "subject":   "How long does shipping take?",
        "content":   "Hey, just wondering how long it usually takes for a package to arrive after I order? Thanks!",
    },
]


async def process_email(email: dict, index: int, app) -> None:
    config = {"configurable": {"thread_id": f"email-{index}"}}

    print(f"\n\nEmail {index}/{len(SAMPLE_EMAILS)}")
    print(f"From: {email['sender']}")
    print(f"Subject: {email['subject']}")
    print(f"Body: {email['content']}")
    print("-" * 60)

    final_reply = None
    async for chunk in app.astream(email, config):
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            first_interrupt = interrupts[0]
            print("Interrupt received.")
            print(f"Interrupt value: {first_interrupt.value}")
            print(f"Interrupt id: {first_interrupt.id}")

            await print_checkpoint(app, config, "After interrupt")

            human_res = reply_tool(email, first_interrupt.value)
            print(f"Resume value: {human_res}")
            print("Resuming graph with Command(resume=...).")

            result = await app.ainvoke(Command(resume=human_res), config)
            final_reply = result.get("auto_reply")

            await print_checkpoint(app, config, "After resume")
            break

        for node_output in chunk.values():
            if isinstance(node_output, dict) and "auto_reply" in node_output:
                final_reply = node_output["auto_reply"]

    if final_reply:
        print("\nFinal auto-reply to sender:")
        print("-" * 60)
        print(final_reply)
        print("=" * 60)

    await print_checkpoint(app, config, "After completion")


async def run_demo(app) -> None:
    tasks = [process_email(email, i, app) for i, email in enumerate(SAMPLE_EMAILS, 1)]
    await asyncio.gather(*tasks)


def main():
    load_env_file()
    if not os.getenv("SARVAM_API_KEY"):
        print("Set SARVAM_API_KEY before running.")
        return

    init_llm_and_classifier()
    app = build_graph()

    workflow_path = os.path.join(os.path.dirname(__file__), "workflow.png")
    write_workflow_png(app, workflow_path)

    print("=" * 60)
    print("        Email Triage Agent - LangGraph Demo")
    print("=" * 60)

    asyncio.run(run_demo(app))


if __name__ == "__main__":
    main()