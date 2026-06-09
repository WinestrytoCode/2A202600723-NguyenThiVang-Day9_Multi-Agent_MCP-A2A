"""
Multi-Agent Supervisor-Workers System for Day 08.

Includes:
  1. Supervisor Node: Routes queries to workers or terminates and synthesizes.
  2. Workers:
     - Legal Penal Worker: Handles criminal code questions (Điều 249, 250, etc.).
     - Rehab & Admin Worker: Handles rehab processes, mandatory/voluntary rehab, and decrees.
     - Showbiz & News Worker: Handles news, celebrity drug cases, and social events.
  3. Synthesizer: Combines all responses into a final cohesive answer.
"""

import os
import json
import logging
from typing import Annotated, TypedDict, List
import operator
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)

# Import retrieval pipeline from task9
from .task9_retrieval_pipeline import retrieve
from .task10_generation import reorder_for_llm, format_context

# ---------------------------------------------------------------------------
# Filtered retrieval helper
# ---------------------------------------------------------------------------
def retrieve_filtered(query: str, doc_type: str, top_k: int = 4) -> list[dict]:
    """Retrieve chunks and filter them by metadata type ('legal' or 'news')."""
    try:
        # Retrieve more chunks to account for filtering
        raw_results = retrieve(query, top_k=top_k * 3)
        filtered = [r for r in raw_results if r.get("metadata", {}).get("type") == doc_type]
        return filtered[:top_k]
    except Exception as e:
        logger.error(f"Error in retrieve_filtered: {e}")
        return []

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class WorkerOutput(TypedDict):
    worker_name: str
    analysis: str
    sources: list[dict]

class AgentState(TypedDict):
    question: str
    history: List[str]
    worker_outputs: Annotated[List[WorkerOutput], operator.add]
    next_worker: str
    final_response: str

# ---------------------------------------------------------------------------
# LLM Caller Helper
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_content: str, temperature: float = 0.3) -> str:
    """Call gpt-4o-mini using LangChain ChatOpenAI."""
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=temperature
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error calling LLM: {e}"

# ---------------------------------------------------------------------------
# Workers Nodes
# ---------------------------------------------------------------------------

def legal_penal_worker(state: AgentState) -> dict:
    """Worker specialized in Penal Law and Criminal drug penalties."""
    print("\n[Worker] Legal Penal Specialist analyzing question...")
    question = state["question"]
    
    # Retrieve relevant legal chunks
    chunks = retrieve_filtered(question, doc_type="legal", top_k=4)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    
    prompt = """Bạn là chuyên gia Luật Hình sự Việt Nam. Hãy phân tích các hành vi vi phạm về tội phạm ma tuý (tàng trữ, vận chuyển, buôn bán) dựa trên luật pháp Việt Nam.
Sử dụng tài liệu trong Context dưới đây để trích dẫn chính xác (ví dụ: [Bộ luật Hình sự 2015, Điều 249]).
Nếu thông tin không có trong Context, hãy chỉ trả lời những gì chắc chắn hoặc ghi nhận giới hạn thông tin.

Context:
{context}
"""
    system_msg = prompt.format(context=context)
    user_msg = f"Hãy phân tích và trả lời câu hỏi sau về luật hình sự ma tuý: {question}"
    
    analysis = call_llm(system_msg, user_msg)
    
    output: WorkerOutput = {
        "worker_name": "Legal Penal Worker",
        "analysis": analysis,
        "sources": chunks
    }
    
    return {
        "worker_outputs": [output],
        "next_worker": "supervisor"
    }


def rehab_admin_worker(state: AgentState) -> dict:
    """Worker specialized in Rehabilitation and Administrative procedures."""
    print("\n[Worker] Rehab & Administration Specialist analyzing question...")
    question = state["question"]
    
    # Retrieve relevant legal chunks (since rehab/admin rules are in legal files)
    chunks = retrieve_filtered(question, doc_type="legal", top_k=4)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    
    prompt = """Bạn là chuyên gia về Quy trình Cai nghiện và Thủ tục Hành chính về ma tuý tại Việt Nam.
Hãy phân tích các khía cạnh liên quan đến cai nghiện bắt buộc, cai nghiện tự nguyện, quản lý sau cai nghiện hoặc xử phạt hành chính dựa trên các tài liệu trong Context.
Trích dẫn chính xác nguồn tài liệu sử dụng (ví dụ: [Nghị định 105/2021, Điều 12] hoặc [Luật Phòng chống ma tuý 2021, Điều 30]).

Context:
{context}
"""
    system_msg = prompt.format(context=context)
    user_msg = f"Hãy phân tích và trả lời câu hỏi sau về cai nghiện và thủ tục hành chính: {question}"
    
    analysis = call_llm(system_msg, user_msg)
    
    output: WorkerOutput = {
        "worker_name": "Rehab & Admin Worker",
        "analysis": analysis,
        "sources": chunks
    }
    
    return {
        "worker_outputs": [output],
        "next_worker": "supervisor"
    }


def showbiz_news_worker(state: AgentState) -> dict:
    """Worker specialized in news, celeb drug cases and social events."""
    print("\n[Worker] Showbiz & Social News Specialist analyzing question...")
    question = state["question"]
    
    # Retrieve relevant news chunks
    chunks = retrieve_filtered(question, doc_type="news", top_k=4)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    
    prompt = """Bạn là chuyên gia về Tin tức Showbiz và Đời sống Xã hội liên quan đến ma tuý.
Hãy trả lời các thông tin liên quan đến các vụ việc thực tế, scandal, nghệ sĩ/người nổi tiếng bị bắt hoặc khởi tố vì ma tuý dựa trên các bài báo trong Context.
Đảm bảo trích dẫn nguồn báo chí (ví dụ: [VnExpress, 2024] hoặc [Báo Tuổi Trẻ, 2023]).
Nếu không tìm thấy tên nghệ sĩ hay sự kiện cụ thể trong Context, ghi rõ 'Tôi không thể xác minh thông tin này từ các bài báo hiện có'.

Context:
{context}
"""
    system_msg = prompt.format(context=context)
    user_msg = f"Hãy cung cấp thông tin xã hội/showbiz dựa trên câu hỏi sau: {question}"
    
    analysis = call_llm(system_msg, user_msg)
    
    output: WorkerOutput = {
        "worker_name": "Showbiz & News Worker",
        "analysis": analysis,
        "sources": chunks
    }
    
    return {
        "worker_outputs": [output],
        "next_worker": "supervisor"
    }

# ---------------------------------------------------------------------------
# Supervisor Node
# ---------------------------------------------------------------------------
def supervisor(state: AgentState) -> dict:
    """Supervisor node deciding which worker should act next or if to finish."""
    print("\n[Supervisor] Evaluating status...")
    
    # Get list of workers who have already run
    already_run = [out["worker_name"] for out in state.get("worker_outputs", [])]
    
    system_prompt = """Bạn là Điều phối viên (Supervisor) của hệ thống Multi-Agent về pháp luật và tin tức ma tuý.
Nhiệm vụ của bạn là đọc câu hỏi của người dùng và quyết định xem chuyên gia nào (Worker) cần được gọi tiếp theo để cung cấp dữ liệu phân tích.

Các chuyên gia khả dụng:
1. "legal_penal_worker": Chuyên về Luật Hình sự Việt Nam, hình phạt, tội phạm ma tuý (tàng trữ, vận chuyển, mua bán).
2. "rehab_admin_worker": Chuyên về Quy trình cai nghiện (bắt buộc/tự nguyện), thủ tục hành chính, nghị định xử phạt hành chính.
3. "showbiz_news_worker": Chuyên về tin tức xã hội, sự kiện bắt bớ thực tế, các nghệ sĩ dính líu đến ma tuý trong showbiz.

Quy tắc điều phối:
- Đánh giá câu hỏi người dùng cần những chuyên môn nào.
- Một câu hỏi phức hợp có thể cần sự tham gia của 2 hoặc cả 3 chuyên gia.
- Không gọi lại chuyên gia đã chạy rồi (danh sách đã chạy: {already_run}).
- Nếu không cần gọi thêm chuyên gia nào nữa (hoặc đã gọi đủ các chuyên gia cần thiết để trả lời câu hỏi), hãy chọn "FINISH".

Định dạng đầu ra: Bạn BẮT BUỘC phải trả về một đối tượng JSON có định dạng chính xác sau đây, không có bất kỳ văn bản nào ngoài JSON:
{{
    "reasoning": "Giải thích ngắn gọn lý do chọn bước đi tiếp theo",
    "next": "tên_worker_hoặc_FINISH"
}}
Tên worker được viết đúng như sau: "legal_penal_worker", "rehab_admin_worker", "showbiz_news_worker", hoặc "FINISH".
"""
    system_msg = system_prompt.format(already_run=", ".join(already_run) if already_run else "Chưa có")
    user_msg = f"Câu hỏi của User: {state['question']}\n\nLịch sử các câu phân tích đã thu thập được: {json.dumps([{ 'worker': o['worker_name'], 'analysis': o['analysis'][:200] + '...' } for o in state.get('worker_outputs', [])], ensure_ascii=False)}"
    
    response = call_llm(system_msg, user_msg, temperature=0.1)
    
    # Parse JSON output
    try:
        data = json.loads(response.strip().replace("```json", "").replace("```", ""))
        next_worker = data.get("next", "FINISH")
        reasoning = data.get("reasoning", "")
        print(f"  Supervisor decided: {next_worker} | Rationale: {reasoning}")
    except Exception as e:
        print(f"  Error parsing supervisor response, defaulting to FINISH. Raw response: {response}")
        next_worker = "FINISH"
        
    return {
        "next_worker": next_worker
    }

# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------
def route_worker(state: AgentState) -> str:
    """Router for conditional edges."""
    return state["next_worker"]

# ---------------------------------------------------------------------------
# Synthesizer Node
# ---------------------------------------------------------------------------
def synthesizer(state: AgentState) -> dict:
    """Synthesizer combining outputs from all consulted workers."""
    print("\n[Synthesizer] Compiling final comprehensive response...")
    question = state["question"]
    outputs = state.get("worker_outputs", [])
    
    if not outputs:
        # If no worker ran, do a direct fallback generation
        print("  No workers were called, performing direct generation...")
        direct_result = retrieve(question, top_k=5)
        reordered = reorder_for_llm(direct_result)
        context = format_context(reordered)
        system_msg = """Bạn là chuyên gia tư vấn luật ma tuý. Hãy trả lời câu hỏi của người dùng dựa vào Context dưới đây. Trích dẫn rõ ràng nguồn tài liệu."""
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        final_answer = call_llm(system_msg, user_msg)
        return {"final_response": final_answer}
    
    # Format worker inputs
    compiled_inputs = []
    for out in outputs:
        compiled_inputs.append(
            f"=== PHÂN TÍCH TỪ {out['worker_name'].upper()} ===\n"
            f"{out['analysis']}\n"
        )
    combined_context = "\n\n".join(compiled_inputs)
    
    system_msg = """Bạn là cố vấn pháp lý tối cao tổng hợp thông tin.
Dưới đây là các phân tích chuyên môn từ các chuyên gia thành viên (Legal Penal, Rehab/Admin, Showbiz/News) về câu hỏi của khách hàng.
Hãy tổng hợp các phân tích này thành một câu trả lời duy nhất, mạch lạc, có cấu trúc rõ ràng, chuyên nghiệp bằng tiếng Việt.
Giữ lại tất cả các trích dẫn nguồn luật ([Nghị định 105/2021], [Bộ luật Hình sự 2015, Điều 249], [VnExpress, 2024], v.v.) từ các phân tích chuyên môn.
Tránh lặp lại thông tin dư thừa giữa các phần.
"""
    user_msg = f"Câu hỏi gốc: {question}\n\nThông tin từ các chuyên gia:\n{combined_context}"
    
    final_answer = call_llm(system_msg, user_msg)
    
    return {
        "final_response": final_answer
    }

# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------
def create_supervisor_graph():
    """Build and compile the Supervisor-Workers StateGraph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("legal_penal_worker", legal_penal_worker)
    workflow.add_node("rehab_admin_worker", rehab_admin_worker)
    workflow.add_node("showbiz_news_worker", showbiz_news_worker)
    workflow.add_node("synthesizer", synthesizer)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edge from supervisor to workers or synthesizer
    workflow.add_conditional_edges(
        "supervisor",
        route_worker,
        {
            "legal_penal_worker": "legal_penal_worker",
            "rehab_admin_worker": "rehab_admin_worker",
            "showbiz_news_worker": "showbiz_news_worker",
            "FINISH": "synthesizer"
        }
    )
    
    # Workers route back to supervisor
    workflow.add_edge("legal_penal_worker", "supervisor")
    workflow.add_edge("rehab_admin_worker", "supervisor")
    workflow.add_edge("showbiz_news_worker", "supervisor")
    
    # Synthesizer leads to END
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# ---------------------------------------------------------------------------
# Runner Helper
# ---------------------------------------------------------------------------
def run_agent(question: str) -> dict:
    """Run the Supervisor-Workers agent end-to-end."""
    graph = create_supervisor_graph()
    initial_state: AgentState = {
        "question": question,
        "history": [],
        "worker_outputs": [],
        "next_worker": "",
        "final_response": ""
    }
    
    result = graph.invoke(initial_state)
    return result

if __name__ == "__main__":
    # Standard quick test queries
    test_queries = [
        "Hình phạt cho tội tàng trữ ma tuý trái phép và quy trình cai nghiện bắt buộc là như thế nào?",
        "Ca sĩ Chi Dân bị bắt liên quan đến ma tuý như thế nào và hình phạt pháp luật cho tội đó là gì?",
    ]
    
    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")
        print(f"{'='*80}")
        res = run_agent(q)
        print(f"\nFINAL ANSWER:\n{res['final_response']}")
        print(f"\nConsulted specialists: {[w['worker_name'] for w in res['worker_outputs']]}")
