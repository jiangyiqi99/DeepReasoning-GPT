import requests
import json
import os
import sys
import time

# API配置
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

CONVERSATION_MAX_HISTORY = 100


class ConversationHistory:
    def __init__(self, max_history=CONVERSATION_MAX_HISTORY):
        """
        初始化对话历史记录类

        Args:
            max_history (int): 保留的最大对话轮次
        """
        self.history = []
        self.max_history = max_history

    def add_interaction(self, user_query, reasoning, ai_response):
        """
        添加一轮完整的交互到历史记录

        Args:
            user_query (str): 用户的问题
            reasoning (str): DeepSeek的推理内容
            ai_response (str): OpenAI的回答
        """
        self.history.append({
            "user_query": user_query,
            "reasoning": reasoning,
            "ai_response": ai_response,
            "timestamp": time.time()
        })

        # 如果历史记录超过最大限制，删除最旧的
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_conversation_for_deepseek(self):
        """
        为DeepSeek构建包含历史上下文的消息列表

        Returns:
            list: 消息列表
        """
        messages = []

        # 添加历史交互
        for interaction in self.history:
            messages.append({
                "role": "user",
                "content": interaction["user_query"]
            })
            messages.append({
                "role": "assistant",
                "content": interaction["ai_response"]
            })

        return messages

    def get_conversation_for_openai(self, current_query, current_reasoning):
        """
        为OpenAI构建包含历史上下文的消息列表

        Args:
            current_query (str): 当前用户的问题
            current_reasoning (str): 当前问题的推理内容

        Returns:
            list: 消息列表
        """
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Use the provided reasoning to answer questions in context of the conversation history."}
        ]

        # 添加历史交互
        for interaction in self.history:
            messages.append({
                "role": "user",
                "content": interaction["user_query"]
            })
            messages.append({
                "role": "assistant",
                "content": interaction["ai_response"]
            })

        # 添加当前问题和推理内容
        messages.append({
            "role": "user",
            "content": f"Answer the following question: {current_query}\n\nAccording to the reasoning prompt: {current_reasoning}\n\nBased on this reasoning and our conversation history, provide your answer to the question."
        })

        return messages

    def get_summary(self):
        """
        获取对话历史的摘要

        Returns:
            str: 历史对话的简短摘要
        """
        if not self.history:
            return "No conversation history yet."

        summary = f"Conversation History ({len(self.history)} interactions):\n"

        for i, interaction in enumerate(self.history, 1):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(interaction["timestamp"]))
            summary += f"\n{i}. {timestamp}\nUser: {interaction['user_query'][:50]}{'...' if len(interaction['user_query']) > 50 else ''}\nAI: {interaction['ai_response'][:50]}{'...' if len(interaction['ai_response']) > 50 else ''}\n"

        return summary


def get_deepseek_reasoning_stream(prompt, conversation_history=None):
    """
    流式调用DeepSeek API并实时输出reasoning_content，
    一旦收集完reasoning_content就终止请求

    Args:
        prompt (str): 用户的原始请求
        conversation_history (ConversationHistory, optional): 对话历史

    Returns:
        str: DeepSeek的完整推理内容
    """
    print("\nDeepSeek reasoning content:")
    sys.stdout.flush()

    # DeepSeek 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # 准备消息，包含历史上下文
    messages = []
    if conversation_history:
        messages = conversation_history.get_conversation_for_deepseek()

    # 添加当前问题
    messages.append({"role": "user", "content": prompt})

    # DeepSeek 请求体
    data = {
        "model": "deepseek-r1-250120",
        "messages": messages,
        "stream": True
    }

    # 发送请求到 DeepSeek API
    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers=headers,
        data=json.dumps(data),
        stream=True
    )

    # 检查响应状态
    if response.status_code != 200:
        print(f"DeepSeek API Error: {response.status_code}")
        print(response.text)
        return None

    # 处理 DeepSeek 的流式响应
    reasoning_content = ""
    reasoning_finished = False
    last_reasoning_content_time = None

    for line in response.iter_lines():
        if not line:
            continue

        # 移除 "data: " 前缀并解析 JSON
        if line.startswith(b'data: '):
            line = line[6:]  # 移除 "data: " 前缀

            # 处理流结束标记
            if line == b'[DONE]':
                break

            try:
                chunk = json.loads(line)

                # 检查是否有finish_reason，表示推理内容已完成
                if 'choices' in chunk and chunk['choices'] and 'finish_reason' in chunk['choices'][0]:
                    if chunk['choices'][0]['finish_reason'] is not None:
                        # 推理已完成，我们可以终止流
                        break

                # 提取 reasoning_content（如果存在）
                if 'choices' in chunk and chunk['choices'] and 'delta' in chunk['choices'][0]:
                    delta = chunk['choices'][0]['delta']

                    # 检查是否有reasoning_content
                    if 'reasoning_content' in delta:
                        # 即使是空内容也会更新最后一次接收时间
                        last_reasoning_content_time = time.time()

                        # 只处理非空内容
                        if delta['reasoning_content']:
                            # 流式输出
                            print(delta['reasoning_content'], end="")
                            sys.stdout.flush()  # 确保立即输出，不缓冲
                            reasoning_content += delta['reasoning_content']

                    # 检查reasoning_content是否已经完成
                    # 如果在delta中看到content而不是reasoning_content，说明已转到正常输出阶段
                    elif 'content' in delta and last_reasoning_content_time is not None:
                        # 已收到过reasoning_content并现在开始收到content，可以终止流
                        reasoning_finished = True
                        break

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    # 如果我们主动终止了响应，需要关闭响应连接
    if reasoning_finished:
        response.close()

    print("\n")  # 在推理内容结束后添加换行
    return reasoning_content


def get_openai_answer_stream(prompt, reasoning_content, conversation_history=None):
    """
    流式调用OpenAI API并实时输出回答

    Args:
        prompt (str): 用户的原始请求
        reasoning_content (str): DeepSeek的推理内容
        conversation_history (ConversationHistory, optional): 对话历史

    Returns:
        str: OpenAI的完整回答
    """
    print("\nGPT-4o Answer:")
    sys.stdout.flush()

    # OpenAI 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # 构建发送给 OpenAI 的消息，包含历史上下文
    if conversation_history:
        messages = conversation_history.get_conversation_for_openai(prompt, reasoning_content)
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"Answer the following question: {prompt}\n\nAccording to the reasoning prompt: {reasoning_content}\n\nBased on this reasoning, provide your answer to the original question."}
        ]

    # OpenAI 请求体 - 启用流式输出
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "stream": True  # 启用流式响应
    }

    # 发送请求到 OpenAI API
    response = requests.post(
        OPENAI_BASE_URL,
        headers=headers,
        data=json.dumps(data),
        stream=True  # 使用流式请求
    )

    # 检查响应状态
    if response.status_code != 200:
        print(f"OpenAI API Error: {response.status_code}")
        print(response.text)
        return None

    # 处理 OpenAI 的流式响应
    complete_answer = ""

    for line in response.iter_lines():
        if not line:
            continue

        # 移除 "data: " 前缀并解析 JSON
        if line.startswith(b'data: '):
            line = line[6:]  # 移除 "data: " 前缀

            # 处理流结束标记
            if line == b'[DONE]':
                break

            try:
                chunk = json.loads(line)

                # 提取内容增量
                if 'choices' in chunk and chunk['choices'] and 'delta' in chunk['choices'][0]:
                    delta = chunk['choices'][0]['delta']

                    if 'content' in delta and delta['content']:
                        # 流式输出
                        print(delta['content'], end="")
                        sys.stdout.flush()  # 确保立即输出，不缓冲
                        complete_answer += delta['content']

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    print("\n")  # 在回答结束后添加换行
    return complete_answer


def process_question(user_prompt, conversation_history):
    """
    处理单个问题的完整流程

    Args:
        user_prompt (str): 用户的问题
        conversation_history (ConversationHistory): 对话历史对象

    Returns:
        bool: 处理是否成功
    """
    print(f"\nQuestion: {user_prompt}")

    # 第一步：获取DeepSeek的推理内容（流式输出）
    reasoning_content = get_deepseek_reasoning_stream(user_prompt, conversation_history)

    if reasoning_content is None:
        print("Failed to get reasoning content from DeepSeek.")
        return False

    # 第二步：获取OpenAI的回答（流式输出）
    gpt4o_answer = get_openai_answer_stream(user_prompt, reasoning_content, conversation_history)

    if gpt4o_answer is None:
        print("Failed to get answer from OpenAI.")
        return False

    # 将当前交互添加到历史记录
    conversation_history.add_interaction(user_prompt, reasoning_content, gpt4o_answer)
    return True


def interactive_mode():
    """
    交互式问答模式，支持上下文对话
    """
    print("Welcome to the AI Assistant with context memory.")
    print("Type '#exit' to end the session.")
    print("Type '#history' to see conversation history.")
    print("Type '#clear' to clear conversation history.")

    # 初始化对话历史记录
    conversation_history = ConversationHistory(max_history = CONVERSATION_MAX_HISTORY)

    while True:
        try:
            # 显示提示符并获取用户输入
            user_input = input("\n>>> ")

            # 检查是否退出
            if user_input.lower() == '#exit':
                print("Goodbye!")
                break#

            # 检查是否请求历史记录
            if user_input.lower() == '#history':
                print(conversation_history.get_summary())
                continue

            # 检查是否清除历史记录
            if user_input.lower() == '#clear':
                conversation_history = ConversationHistory(max_history=CONVERSATION_MAX_HISTORY)
                # Windows 系统
                if os.name == 'nt':
                    os.system('cls')
                # Mac 和 Linux 系统
                else:
                    os.system('clear')
                continue

            # 处理空输入
            if not user_input.strip():
                continue

            # 处理用户问题
            process_question(user_input, conversation_history)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Let's continue with a new question.")


def main():
    """
    主函数，启动交互式问答模式
    """
    print("Starting AI Assistant with context memory...")
    interactive_mode()


if __name__ == "__main__":
    main()
