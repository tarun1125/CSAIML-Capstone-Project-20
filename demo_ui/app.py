"""
Live demo UI for the capstone defense: Baseline vs RAG vs Fine-Tuned Qwen2.5-Coder-1.5B
on NL-to-PyMongo query generation.

Run from the repo root (needs project .venv active for CURATED mode's dependency-free
tabs to still just work with plain streamlit; LIVE mode additionally needs mlx_lm,
faiss, sentence-transformers/torch, and pymongo -- see demo_ui/README.md):

    source .venv/bin/activate
    streamlit run demo_ui/app.py

Design intent (right-sized for a defense demo, not over-engineered):
  - CURATED mode (default) replays 5 real, source-verified examples with zero
    model loading and zero network calls -- this is what you present live, because
    it can never fail, hang, or produce a surprising answer mid-defense.
  - LIVE mode is an opt-in "ask it something new" capability for Q&A -- it actually
    loads Qwen + the relevant adapter and generates on the spot. It's slower (model
    cold-start) and depends on the machine having mlx_lm installed, so it's presented
    as a deliberate bonus, not the backbone of the demo.
  - Optional "execute against Atlas" is off by default in both modes and clearly
    labeled, since it needs live DB credentials/network and isn't needed to make
    the core point (query correctness is usually visible by inspection for these
    example questions).
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo_ui.app")

from curated_examples import CURATED_EXAMPLES, get_example_by_id  # noqa: E402

st.set_page_config(page_title="NL-to-MongoDB: Baseline vs RAG vs Fine-Tuned", layout="wide")


def status_badge(execution_accuracy: bool) -> str:
    return "✅ CORRECT (matches gold)" if execution_accuracy else "❌ WRONG / FAILED"


def render_arm_result(arm_name: str, arm_data: dict, gold: dict):
    st.markdown(f"**{arm_name}**")
    st.code(arm_data["query"], language="python")
    st.caption(f"Execution status: `{arm_data['status']}`")
    st.markdown(status_badge(arm_data["execution_accuracy"]))
    with st.expander("Result returned"):
        st.json(arm_data["result"])
    if arm_data.get("note"):
        st.info(arm_data["note"])


def render_single_arm_tab(arm_key: str, arm_label: str):
    st.subheader(f"{arm_label} — curated examples")
    example_labels = {ex["id"]: f"{ex['id']} — {ex['question'][:60]}" for ex in CURATED_EXAMPLES}
    selected_id = st.selectbox(
        "Pick an example", options=list(example_labels.keys()),
        format_func=lambda i: example_labels[i], key=f"select_{arm_key}",
    )
    example = get_example_by_id(selected_id)
    if example is None:
        st.error(f"Example {selected_id} not found.")
        return

    st.markdown(f"**Question:** {example['question']}")
    st.markdown(f"**Database:** `{example['database']}`")
    st.markdown(f"**Demonstrated pattern:** {example['pattern']}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_arm_result(arm_label, example[arm_key], example["gold"])
    with col2:
        st.markdown("**Gold (ground truth)**")
        st.code(example["gold"]["query"], language="python")
        with st.expander("Gold result"):
            st.json(example["gold"]["result"])

    logger.info("Rendered curated example id=%s arm=%s status=%s", selected_id, arm_key,
                example[arm_key]["status"])


def render_compare_all_tab():
    st.subheader("Compare all 3 arms side by side")
    example_labels = {ex["id"]: f"{ex['id']} — {ex['question'][:60]} [{ex['pattern']}]" for ex in CURATED_EXAMPLES}
    selected_id = st.selectbox(
        "Pick an example", options=list(example_labels.keys()),
        format_func=lambda i: example_labels[i], key="select_compare",
    )
    example = get_example_by_id(selected_id)
    if example is None:
        st.error(f"Example {selected_id} not found.")
        return

    st.markdown(f"**Question:** {example['question']}  ·  **Database:** `{example['database']}`")
    st.caption(example["pattern"])
    st.divider()

    cols = st.columns(4)
    arm_order = [("baseline", "Baseline (zero-shot)"), ("rag", "RAG"), ("finetuned", "Fine-Tuned (LoRA)")]
    for col, (arm_key, arm_label) in zip(cols[:3], arm_order):
        with col:
            render_arm_result(arm_label, example[arm_key], example["gold"])
    with cols[3]:
        st.markdown("**Gold**")
        st.code(example["gold"]["query"], language="python")
        with st.expander("Gold result"):
            st.json(example["gold"]["result"])

    st.divider()
    summary_cols = st.columns(3)
    for col, (arm_key, arm_label) in zip(summary_cols, arm_order):
        with col:
            ok = example[arm_key]["execution_accuracy"]
            (st.success if ok else st.error)(f"{arm_label}: {'PASS' if ok else 'FAIL'}")

    logger.info("Rendered compare-all view for id=%s", selected_id)


def render_live_tab():
    st.subheader("Live inference (experimental)")
    st.warning(
        "This calls the real model on this machine via MLX. It requires mlx_lm (and, for RAG, faiss + "
        "sentence-transformers/torch) to be importable from whatever Python is running Streamlit — i.e. "
        "run this app from the project's own `.venv`. First generation per arm is slow (cold model load). "
        "Recommended: rehearse this once before the actual defense rather than trying it live for the first time."
    )

    try:
        import live_inference as li
    except ImportError as exc:
        st.error(f"live_inference module failed to import: {exc}")
        logger.exception("live_inference import failed")
        return

    question = st.text_area("Question (natural language)", value="How many rows are in the largest table?")
    database = st.text_input("Database name", value="car_1")
    arm = st.radio("Arm to run", ["baseline", "rag", "finetuned"], horizontal=True)

    ft_variant = None
    if arm == "finetuned":
        ft_variant = st.selectbox(
            "Fine-tuned adapter variant",
            options=["finetuned_23db_1000iter", "finetuned_23db_200iter", "finetuned_6db"],
            help="finetuned_23db_1000iter is the current best (epoch-parity-fixed) adapter. "
                 "23db variants only work for the 23 databases they were trained on.",
        )

    run_atlas = st.checkbox("Also execute the generated query against live Atlas (needs atlas-credentials.env)",
                             value=False)

    if st.button("Generate", type="primary"):
        logger.info("Live-inference button pressed: arm=%s database=%s variant=%s", arm, database, ft_variant)
        with st.spinner("Loading model / generating (first call per arm can take a while)..."):
            try:
                if arm == "baseline":
                    result = li.run_baseline(question, database)
                elif arm == "rag":
                    result = li.run_rag(question, database)
                else:
                    result = li.run_finetuned(question, database, variant=ft_variant)
            except li.ModelUnavailable as exc:
                st.error(str(exc))
                logger.warning("Live inference unavailable: %s", exc)
                return
            except Exception as exc:  # noqa: BLE001 -- surface any failure in the UI, never crash the app
                st.error(f"Generation failed: {exc}")
                logger.exception("Live inference failed unexpectedly")
                return

        st.markdown("**Generated query:**")
        st.code(result["generated_query"], language="python")
        if "predicted_database" in result:
            st.caption(f"RAG's retrieval predicted database: `{result['predicted_database']}` "
                       f"(you specified `{database}`) -- RAG never sees your database field, only the question.")
        if "intended_database" in result:
            st.caption(
                f"You intended `{result['intended_database']}`. Baseline is shown the schemas of ALL "
                f"test-slice databases at once (this matches the real benchmark condition, not a "
                f"simplified single-DB prompt) -- check which collection name appears in the generated "
                f"query above to see which database the model actually picked. Vague, domain-agnostic "
                f"questions (like 'how many rows in the largest table') give it no way to choose "
                f"correctly; specific, entity-naming questions (like the curated examples) work far "
                f"better here."
            )
        with st.expander("Full result payload"):
            st.json(result)

        if run_atlas:
            with st.spinner("Executing against Atlas..."):
                exec_result = li.execute_against_atlas(result["generated_query"], database)
            if exec_result["status"] == "PASS":
                st.success("Query executed successfully.")
                st.json(exec_result["result"])
            else:
                st.error(f"Execution failed: {exec_result['error']}")


def main():
    st.title("NL-to-MongoDB Query Generation: Baseline vs RAG vs Fine-Tuned")
    st.caption(
        "IIIT capstone defense demo. Curated examples below are pulled verbatim from this repo's own "
        "execution-result JSON files (baseline/RAG scored on the 304-case holdout, fine-tuned on the "
        "epoch-parity-fixed 23-db adapter, 1000 iterations / ~3.7 epochs) — nothing here is fabricated."
    )

    tab_baseline, tab_rag, tab_ft, tab_compare, tab_live = st.tabs(
        ["Baseline", "RAG", "Fine-Tuned", "Compare All 3", "Live Inference (experimental)"]
    )
    with tab_baseline:
        render_single_arm_tab("baseline", "Baseline (zero-shot)")
    with tab_rag:
        render_single_arm_tab("rag", "RAG")
    with tab_ft:
        render_single_arm_tab("finetuned", "Fine-Tuned (LoRA, 23db, 1000 iters)")
    with tab_compare:
        render_compare_all_tab()
    with tab_live:
        render_live_tab()


if __name__ == "__main__":
    logger.info("demo_ui/app.py starting, %d curated examples loaded", len(CURATED_EXAMPLES))
    main()
