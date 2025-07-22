from ..Utils.utils import print_info, print_limited_json
from agent.Prompt import *
from agent.LLM import *
from .action import *
import time
import json5
from .action import ResponseError
from logs import logger

#RAG logger
from ..Utils.rag_logger import RAGLogger
import copy

# Additional imports for operator support
from agent.Prompt.prompt_constructor import OperatorPromptConstructor, OperatorPromptRAGConstructor
from agent.Utils.utils import is_valid_base64
import json
from typing import List, Dict, Any
import os

class InteractionMode:
    def __init__(self, text_model=None, visual_model=None):
        self.text_model = text_model
        self.visual_model = visual_model

    def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        # Returns a six-tuple containing None, consistent with DomMode
        return None, None, None, None, None, None

class DomMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)
    
    async def execute(self, status_description, user_request, rag_enabled,rag_path, previous_trace, observation, feedback, observation_VforD):
        rag_data = {
            "rag_enabled": rag_enabled,
            "rag_path": rag_path if rag_enabled else None,
        }

        if rag_enabled:
            prompt_constructor = PlanningPromptDescriptionRetrievalConstructor()
            # PlanningPromptVisionRetrievalConstructor 
            
            rag_data["rag_method"] = prompt_constructor.__class__.__name__
            # planning_request
            planning_request = prompt_constructor.construct(
                user_request, rag_path, previous_trace, observation, feedback,  status_description)

            # Record the retrieved example information (from prompt_constructor)
            if hasattr(prompt_constructor, 'reference') and prompt_constructor. reference:
                rag_data["retrieved_examples"] = prompt_constructor.reference
        else:
            planning_request = PlanningPromptConstructor().construct(user_request, previous_trace, observation, feedback, status_description)

        planning_request_copy = copy.deepcopy(planning_request)
        rag_data["planning_request"] = planning_request_copy

        logger.info(
            f"\033[32mDOM_based_planning_request:\n{planning_request}\033[0m\n")
        logger.info(f"planning_text_model: {self.text_model.model}")
        planning_response, error_message = await self.text_model.request(planning_request)

        # Logging the response
        rag_data["planning_response"] = planning_response

        input_token_count = calculation_of_token(planning_request, model=self.text_model.model)
        output_token_count = calculation_of_token(planning_response, model=self.text_model.model)
        planning_token_count = [input_token_count, output_token_count]

        rag_data["token_counts"] = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count
        }

        return planning_response, error_message, None, None, planning_token_count, rag_data

class DomVDescMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)

    async def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        if observation_VforD != "":
            vision_desc_request = VisionDisc2PromptConstructor().construct(
                user_request, observation_VforD)  # vision description request with user_request
            # vision_desc_request = VisionDisc1PromptConstructor().construct(observation_VforD)
            vision_desc_response, error_message = await self.visual_model.request(vision_desc_request)
        else:
            vision_desc_response = ""
        print(f"\033[36mvision_disc_response:\n{vision_desc_response}")  # blue
        planning_request = ObservationVisionDiscPromptConstructor().construct(
            user_request, previous_trace, observation, feedback, status_description, vision_desc_response)
        print(
            f"\033[35mplanning_request:\n{print_limited_json(planning_request, limit=10000)}")
        print("\033[0m")
        planning_response, error_message = await self.text_model.request(planning_request)
        return planning_response, error_message, None, None


class VisionToDomMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)

    async def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        vision_act_request = ObservationVisionActPromptConstructor().construct(
            user_request, previous_trace, observation_VforD, feedback, status_description)
        max_retries = 3
        for attempt in range(max_retries):
            vision_act_response, error_message = await self.visual_model.request(vision_act_request)
            # Blue output
            print(f"\033[36mvision_act_response:\n{vision_act_response}")
            print("\033[0m")  # Reset color
            planning_response_thought, planning_response_get = ActionParser().extract_thought_and_action(
                vision_act_response)
            actions = {
                'goto': "Found 'goto' in the vision_act_response.",
                # 'google_search': "Found 'google_search' in the vision_act_response.",
                'switch_tab': "Found 'switch_tab' in the vision_act_response.",
                'scroll_down': "Found 'scroll_down' in the vision_act_response.",
                'scroll_up': "Found 'scroll_up' in the vision_act_response.",
                'go_back': "Found 'go_back' in the vision_act_response."
            }
            # Check if the action is in the predefined action list
            actions_found = False
            for action, message in actions.items():
                if action == planning_response_get.get('action'):
                    print(message)
                    actions_found = True
                    # The action does not need to be changed
                    # `target_element` should not exist, if it does, it's not used
                    break

            if not actions_found:
                # print("None of 'goto', 'google_search', 'switch_tab', 'scroll_down', 'scroll_up', or 'go_back' were found in the vision_act_response.")
                print("None of 'goto', 'switch_tab', 'scroll_down', 'scroll_up', or 'go_back' were found in the vision_act_response.")

                target_element = planning_response_get.get('target_element')
                description = planning_response_get.get('description')

                # If the target element is None or does not exist
                if not target_element:
                    print("The 'target_element' is None or empty.")
                    continue

                # Construct the request from vision to DOM
                planning_request = VisionToDomPromptConstructor().construct(target_element, description,
                                                                            observation)
                print(f"\033[35mplanning_request:{planning_request}")
                print("\033[0m")

                # Send the request and wait for the response
                planning_response_dom, error_message = await self.text_model.request(planning_request)
                print(
                    f"\033[34mVisionToDomplanning_response:\n{planning_response_dom}")
                print("\033[0m")
                # Parse the element ID
                element_id = ActionParser().get_element_id(planning_response_dom)
                if element_id == "-1":
                    print("The 'element_id' is not found in the planning_response.")
                    continue  # If the 'element_id' is not found, continue to the next iteration of the loop
                else:
                    planning_response_get['element_id'] = element_id
                    break  # If the 'element_id' is found, break the loop

            else:
                # If a predefined action is found, there is no need to retry, exit the loop directly
                break

        planning_response_json_str = json5.dumps(
            planning_response_get, indent=2)
        planning_response = f'```\n{planning_response_json_str}\n```'
        # Check if the maximum number of retries has been reached
        if attempt == max_retries - 1:
            print("Max retries of vision_act reached. Unable to proceed.")

        return planning_response, error_message, planning_response_thought, planning_response_get


class DVMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)

    async def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        planning_request = D_VObservationPromptConstructor().construct(
            user_request, previous_trace, observation, observation_VforD, feedback, status_description)

        print(
            f"\033[32mplanning_request:\n{print_limited_json(planning_request, limit=1000)}")
        print("\033[0m")
        planning_response, error_message = await self.visual_model.request(planning_request)
        return planning_response, error_message, None, None


class VisionMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)

    async def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        planning_request = VisionObservationPromptConstructor(
        ).construct(user_request, previous_trace, observation)
        print(f"\033[32m{planning_request}")  # Green color
        print("\033[0m")
        logger.info("\033[32m%s\033[0m", planning_request)
        planning_response, error_message = await self.visual_model.request(planning_request)
        return planning_response, error_message, None, None


class OperatorMode(InteractionMode):
    """
    OpenAI Operator mode for browser automation
    """
    
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)
        # Operator is the primary model for this mode
        self.operator_model = text_model
        self.conversation_history = []
        self.current_screenshot = None
    
    async def execute(self, status_description, user_request, rag_enabled, rag_path, 
                     previous_trace, observation, feedback, observation_VforD):
        """
        Execute operator planning with proper OpenAI Operator integration
        
        Args:
            status_description: Current task status
            user_request: User's task request
            rag_enabled: Whether RAG is enabled
            rag_path: Path to RAG data
            previous_trace: Previous action history
            observation: Current DOM observation (not used in operator mode)
            feedback: Any feedback or error messages
            observation_VforD: Screenshot in base64 format
            
        Returns:
            Tuple of (planning_response, error_message, planning_response_thought, 
                     planning_response_action, planning_token_count, rag_data)
        """
        rag_data = {
            "rag_enabled": rag_enabled,
            "rag_path": rag_path if rag_enabled else None,
            "mode": "operator"
        }
        
        # 初始化current_rag_data用于记录详细的RAG信息
        self.current_rag_data = rag_data.copy()
        
        try:
            # Store current screenshot for conversation continuity
            self.current_screenshot = observation_VforD
            
            # Build conversation messages
            messages = self._build_operator_messages(user_request, status_description, 
                                                   previous_trace, feedback, rag_enabled, rag_path)
            
            # 将构建过程中收集的RAG信息合并到最终的rag_data中
            rag_data.update(self.current_rag_data)
            
            # Log the planning request
            logger.info(f"\033[36mOperator Planning Request:\n{user_request}\033[0m")
            logger.info(f"Operator Model: {self.operator_model.model}")
            logger.info(f"Screenshot Available: {observation_VforD is not None}")
            if rag_enabled:
                logger.info(f"🧠 RAG Mode: {rag_data.get('rag_constructor_type', 'Unknown')}")
                logger.info(f"📚 RAG Reference Available: {bool(rag_data.get('rag_reference', ''))}")
            
            # Make request to OpenAI Operator
            planning_response, error_message = await self.operator_model.request(
                messages=messages,
                screenshot_base64=observation_VforD,
                viewport_width=1280,
                viewport_height=720
            )
            
            if error_message:
                logger.error(f"Operator API Error: {error_message}")
                rag_data["error"] = error_message
                return ("", error_message, "", {"action": "wait", "action_input": "1"}, [0, 0], rag_data)
            
            # Log the response
            logger.info(f"\033[32mOperator Response:\n{planning_response}\033[0m")
            rag_data["planning_response"] = planning_response
            
            # Parse the operator response
            try:
                response_data = json.loads(planning_response)
                actions = response_data.get("actions", [])
                text_response = response_data.get("text_response", "")
                reasoning = response_data.get("reasoning", "")
                
                # Extract thought and action for compatibility with existing code
                # Use text_response as the thought content, no need to parse JSON from it
                planning_response_thought = text_response or reasoning or "Operator processing..."
                
                # Convert operator actions to compatible format
                if actions:
                    # Take the first action for now
                    first_action = actions[0]
                    action_data = first_action.get("action", {})
                    action_type = action_data.get("type", "wait")
                    
                    planning_response_action = self._convert_operator_action(action_data)
                else:
                    # No actions returned, try to parse action from text_response
                    planning_response_action = self._parse_action_from_text(text_response)
                
                # Calculate token counts (approximate)
                input_token_count = len(str(messages)) // 4  # Rough estimate
                output_token_count = len(planning_response) // 4
                planning_token_count = [input_token_count, output_token_count]
                
                # Store in conversation history for continuity
                self.conversation_history.append({
                    "messages": messages,
                    "response": planning_response,
                    "actions": actions
                })
                
                return (planning_response, error_message, planning_response_thought, 
                       planning_response_action, planning_token_count, rag_data)
                
            except Exception as e:
                logger.error(f"Error parsing operator response: {e}")
                planning_response_thought = "Error parsing operator response"
                planning_response_action = {"action": "wait", "action_input": "1", "element_id": "error_wait"}
                return (planning_response, str(e), planning_response_thought, 
                       planning_response_action, [0, 0], rag_data)
            
        except Exception as e:
            logger.error(f"Error in OperatorMode.execute: {e}")
            error_message = str(e)
            return ("", error_message, "", {"action": "wait", "action_input": "1"}, [0, 0], rag_data)
    
    def _build_operator_messages(self, user_request: str, status_description: str,
                               previous_trace: str, feedback: str, 
                               rag_enabled: bool, rag_path: str) -> List[Dict[str, Any]]:
        """
        Build messages for OpenAI Operator
        
        Args:
            user_request: User's task request
            status_description: Current task status
            previous_trace: Previous action history
            feedback: Any feedback messages
            rag_enabled: Whether RAG is enabled
            rag_path: Path to RAG data
            
        Returns:
            Formatted messages for operator
        """
        if rag_enabled:
            # 使用基于描述的RAG构造器
            from agent.Prompt.prompt_constructor import OperatorPromptDescriptionRetrievalConstructor
            
            rag_constructor = OperatorPromptDescriptionRetrievalConstructor()
            
            if hasattr(self, 'current_rag_data'):
                self.current_rag_data["rag_method"] = rag_constructor.__class__.__name__
                self.current_rag_data["rag_constructor_type"] = "OperatorPromptDescriptionRetrievalConstructor"
            
            previous_trace_list = []
            if previous_trace:
                # 简单解析previous_trace字符串
                trace_lines = previous_trace.split('\n')
                for line in trace_lines:
                    if line.strip():
                        # 尝试解析 "Step X: thought -> action" 格式
                        if " -> " in line:
                            parts = line.split(" -> ")
                            if len(parts) >= 2:
                                thought_part = parts[0].strip()
                                action_part = parts[1].strip()
                                
                                # 提取思考内容（去掉 "Step X: " 前缀）
                                if ":" in thought_part:
                                    thought = thought_part.split(":", 1)[1].strip()
                                else:
                                    thought = thought_part
                                
                                previous_trace_list.append({
                                    "thought": thought,
                                    "action": action_part,
                                    "reflection": ""
                                })
                        else:
                            # 格式不匹配:创建一个基本的trace项
                            previous_trace_list.append({
                                "thought": "Previous action",
                                "action": line.strip(),
                                "reflection": ""
                            })
            
            # RAG messages
            messages = rag_constructor.construct(
                user_request=user_request,
                rag_path=rag_path,
                previous_trace=previous_trace_list,
                observation="",  # Operator模式不使用DOM观察
                feedback=feedback,
                status_description=status_description,
                screenshot_base64=self.current_screenshot
            )
            
            if hasattr(self, 'current_rag_data'):
                self.current_rag_data.update({
                    "rag_constructor_used": True,
                    "rag_reference": getattr(rag_constructor, 'reference', ""),
                    "previous_trace_processed": previous_trace_list,
                    "messages_count": len(messages),
                    "user_request": user_request,
                    "status_description": status_description,
                    "feedback": feedback,
                    "screenshot_available": self.current_screenshot is not None
                })
                
                # 记录构建的prompt内容（去除图片数据）
                safe_messages = []
                for msg in messages:
                    safe_msg = {"role": msg["role"]}
                    if msg["role"] == "system":
                        safe_msg["content"] = msg["content"]
                    elif msg["role"] == "user" and isinstance(msg["content"], list):
                        safe_content = []
                        for item in msg["content"]:
                            if item["type"] == "input_text":
                                safe_content.append({
                                    "type": "input_text",
                                    "text": item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"]
                                })
                            elif item["type"] == "input_image":
                                safe_content.append({
                                    "type": "input_image",
                                    "image_url": "[base64_image_data_removed]"
                                })
                        safe_msg["content"] = safe_content
                    else:
                        safe_msg["content"] = str(msg["content"])[:500] + "..." if len(str(msg["content"])) > 500 else str(msg["content"])
                    safe_messages.append(safe_msg)
                
                self.current_rag_data["constructed_messages"] = safe_messages
            
            logger.info(f"🧠 RAG Constructor: {rag_constructor.__class__.__name__}")
            logger.info(f"📚 RAG Reference Length: {len(getattr(rag_constructor, 'reference', ''))}")
            
            return messages
        else:
            # Original simple message construction logic
            if hasattr(self, 'current_rag_data'):
                self.current_rag_data.update({
                    "rag_constructor_used": False,
                    "simple_message_construction": True
                })
            
            messages = []
            
            # System message for operator
            system_message = """You are OpenAI Operator, an AI agent specialized in browser automation.

Your primary objective is to complete web-based tasks efficiently and accurately by analyzing screenshots and providing specific actions.

## Core Capabilities:
- Visual analysis of webpage screenshots
- Understanding of web UI elements and their interactions
- Precise action planning based on current page state
- Step-by-step task completion

## Action Types Available:
1. **click**: Click on buttons, links, or interactive elements at specific coordinates
2. **double_click**: Double-click at specific coordinates
3. **type**: Input text into form fields, search boxes, or text areas
4. **scroll**: Scroll up/down or left/right on the page
5. **keypress**: Press specific keys (Enter, Escape, Tab, etc.)
6. **drag**: Drag from one point to another
7. **wait**: Wait for a specified time (in milliseconds)

## Guidelines:
- Analyze the screenshot carefully before deciding on actions
- Be precise with coordinate selection
- Consider the user's ultimate goal when planning each step
- If you're unsure about an element, explain your reasoning
- Always provide clear feedback about what you're doing and why

## Current Task Context:
"""
            
            if status_description:
                system_message += f"\n**Task Status**: {status_description}"
            
            if previous_trace:
                system_message += f"\n**Previous Actions**: {previous_trace}"
            
            if feedback:
                system_message += f"\n**Feedback**: {feedback}"
            
            messages.append({"role": "system", "content": system_message})
            
            # User message with task request
            user_message = f"""## Current Task:
{user_request}

## Instructions:
1. Analyze the provided screenshot carefully
2. Identify the next logical action to progress toward completing the task
3. Provide precise coordinates for any click/drag actions
4. Explain your reasoning for the chosen action
5. If the task appears complete, indicate this clearly

Please analyze the current state and provide your next action."""
            
            messages.append({"role": "user", "content": user_message})
            
            return messages
    
    def _convert_operator_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI Operator action to compatible format for existing code
        
        Args:
            action_data: Raw action data from operator
            
        Returns:
            Compatible action format
        """
        action_type = action_data.get("type", "wait")
        
        if action_type == "click":
            return {
                "action": "operator_click",
                "action_input": f"{action_data.get('x', 0)},{action_data.get('y', 0)}",
                "coordinates": [action_data.get('x', 0), action_data.get('y', 0)],
                "element_id": f"coord_{action_data.get('x', 0)}_{action_data.get('y', 0)}"
            }
        elif action_type == "double_click":
            return {
                "action": "operator_double_click",
                "action_input": f"{action_data.get('x', 0)},{action_data.get('y', 0)}",
                "coordinates": [action_data.get('x', 0), action_data.get('y', 0)],
                "element_id": f"coord_{action_data.get('x', 0)}_{action_data.get('y', 0)}"
            }
        elif action_type == "type":
            return {
                "action": "operator_type",
                "action_input": action_data.get("text", ""),
                "text": action_data.get("text", ""),
                "element_id": "text_input"
            }
        elif action_type == "scroll":
            return {
                "action": "operator_scroll",
                "action_input": f"{action_data.get('scroll_x', 0)},{action_data.get('scroll_y', 0)}",
                "scroll_x": action_data.get('scroll_x', 0),
                "scroll_y": action_data.get('scroll_y', 0),
                "element_id": "scroll_action"
            }
        elif action_type == "keypress":
            keys = action_data.get("keys", [])
            return {
                "action": "operator_keypress",
                "action_input": ",".join(keys),
                "keys": keys,
                "element_id": "keypress_action"
            }
        elif action_type == "drag":
            path = action_data.get("path", [[0, 0], [0, 0]])
            return {
                "action": "operator_drag",
                "action_input": f"{path[0][0]},{path[0][1]}-{path[-1][0]},{path[-1][1]}",
                "path": path,
                "element_id": f"drag_{path[0][0]}_{path[0][1]}_to_{path[-1][0]}_{path[-1][1]}"
            }
        elif action_type == "wait":
            return {
                "action": "operator_wait",
                "action_input": str(action_data.get("ms", 1000)),
                "ms": action_data.get("ms", 1000),
                "element_id": "wait_action"
            }
        else:
            return {
                "action": "wait",
                "action_input": "1",
                "element_id": "default_wait"
            }
    
    def _parse_action_from_text(self, text_response: str) -> Dict[str, Any]:
        """
        Attempt to parse an action from a text response.
        This handles cases where the operator returns action info in text_response instead of actions array.
        """
        try:
            # Try to find JSON in the text response
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                action_data = json.loads(json_str)
                
                # Extract coordinates and action type
                action_type = action_data.get("action", "click")
                action_input = action_data.get("action_input", {})
                
                if action_type == "click" and isinstance(action_input, dict):
                    x = action_input.get("x", 0)
                    y = action_input.get("y", 0)
                    return {
                        "action": "operator_click",
                        "action_input": f"{x},{y}",
                        "coordinates": [x, y],
                        "element_id": f"coord_{x}_{y}"
                    }
                elif action_type == "type" and isinstance(action_input, dict):
                    text = action_input.get("text", "")
                    return {
                        "action": "operator_type",
                        "action_input": text,
                        "text": text,
                        "element_id": "text_input"
                    }
                # Add more action types as needed
                
        except Exception as e:
            logger.warning(f"Could not parse JSON from text_response: {e}")
        
        # Fallback to simple text analysis
        text_lower = text_response.lower()
        if "click" in text_lower:
            # Try to extract coordinates from text
            import re
            coord_match = re.search(r'"x":\s*(\d+).*?"y":\s*(\d+)', text_response)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                return {
                    "action": "operator_click",
                    "action_input": f"{x},{y}",
                    "coordinates": [x, y],
                    "element_id": f"coord_{x}_{y}"
                }
            return {"action": "operator_click", "action_input": "640,360", "coordinates": [640, 360], "element_id": "default_click"}
        elif "type" in text_lower:
            return {"action": "operator_type", "action_input": "", "text": "", "element_id": "type_action"}
        elif "scroll" in text_lower:
            return {"action": "operator_scroll", "action_input": "0,100", "scroll_x": 0, "scroll_y": 100, "element_id": "scroll_action"}
        elif "wait" in text_lower:
            return {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "wait_action"}
        else:
            return {"action": "wait", "action_input": "1000", "ms": 1000, "element_id": "default_wait"}
    
    def _load_rag_context(self, rag_path: str) -> str:
        """
        Load RAG context from file
        
        Args:
            rag_path: Path to RAG data file
            
        Returns:
            RAG context string
        """
        try:
            if os.path.exists(rag_path):
                with open(rag_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading RAG context: {e}")
        return ""
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history.clear()
        self.current_screenshot = None
        if hasattr(self.operator_model, 'reset_conversation'):
            self.operator_model.reset_conversation()


class Planning:

    @staticmethod
    async def plan(
        config,
        user_request,
        text_model_name,
        previous_trace,
        observation,
        feedback,
        mode,
        observation_VforD,
        status_description,
        rag_enabled,
        rag_path
    ):

        rag_logger = RAGLogger()
        # # Get the current step index from previous_trace
        step_idx = len(previous_trace)
        # task id
        task_id = f"{user_request[:50]}_{int(time.time())}"
        if hasattr(config, 'task_id') and config.task_id:
            task_id = config.task_id
        
        gpt35 = GPTGenerator(model="gpt-3.5-turbo")
        gpt4v = GPTGenerator(model="gpt-4-turbo")

        all_json_models = config["model"]["json_models"]
        is_json_response = config["model"]["json_model_response"]

        llm_planning_text = create_llm_instance(
            text_model_name, is_json_response, all_json_models)

        modes = {
            "dom": DomMode(text_model=llm_planning_text),
            "dom_v_desc": DomVDescMode(visual_model=gpt4v, text_model=llm_planning_text),
            "vision_to_dom": VisionToDomMode(visual_model=gpt4v, text_model=llm_planning_text),
            "d_v": DVMode(visual_model=gpt4v),
            "vision": VisionMode(visual_model=gpt4v),
            "operator": OperatorMode(text_model=llm_planning_text)  # Add operator mode
        }

        result = await modes[mode].execute(
            status_description=status_description,
            user_request=user_request,
            rag_enabled=rag_enabled,
            rag_path=rag_path,
            previous_trace=previous_trace,
            observation=observation,
            feedback=feedback,
            observation_VforD=observation_VforD)
        
        # Check if any RAG data is returned
        if len(result) >= 6 and mode in ["dom", "operator"]:  # Both DomMode and OperatorMode return rag_data
            planning_response, error_message, planning_response_thought, planning_response_action, planning_token_count, rag_data = result
        
            rag_data["mode"] = mode
            rag_data["user_request"] = user_request
            rag_logger.log_rag_step(task_id, step_idx, rag_data)
        else:
            # Compatible with other patterns that do not return rag_data
            planning_response, error_message, planning_response_thought, planning_response_action, planning_token_count = result
        
            # log
            rag_data = {
                "mode": mode,
                "user_request": user_request,
                "rag_enabled": False
            }
            rag_logger.log_rag_step(task_id, step_idx, rag_data)

        logger.info(f"\033[34mPlanning_Response:\n{planning_response}\033[0m")
        
        # Handle operator mode differently - it already parses the response
        if mode != "vision_to_dom" and mode != "operator":
            try:
                planning_response_thought, planning_response_action = ActionParser().extract_thought_and_action(
                    planning_response)
            except ResponseError as e:
                logger.error(f"Response Error:{e.message}")
                raise

        # Special handling for fill_form -> fill_search conversion
        if planning_response_action.get('action') == "fill_form":
            JudgeSearchbarRequest = JudgeSearchbarPromptConstructor().construct(
                input_element=observation, planning_response_action=planning_response_action)
            try:
                Judge_response, error_message = await gpt35.request(JudgeSearchbarRequest)
                if Judge_response.lower() == "yes":
                    planning_response_action['action'] = "fill_search"
            except:
                planning_response_action['action'] = "fill_form"

        # The description should include both the thought (returned by LLM) and the action (parsed from the planning response)
        planning_response_action["description"] = {
            "thought": planning_response_thought,
            "action": (
                f'{planning_response_action["action"]}: {planning_response_action["action_input"]}' if "description" not in planning_response_action.keys() else
                planning_response_action["description"])
            if mode in ["dom","d_v", "dom_v_desc", "vision_to_dom", "operator"] else (
                planning_response_action["action"] if "description" not in planning_response_action.keys() else
                planning_response_action["description"])
        }
        
        # Format action based on mode
        if mode in ["dom", "d_v", "dom_v_desc", "vision_to_dom", "operator"]:
            planning_response_action = {element: planning_response_action.get(
                element, "") for element in ["element_id", "action", "action_input", "description"]}
        elif mode == "vision":
            planning_response_action = {element: planning_response_action.get(
                element, "") for element in ["action", "description"]}
        
        logger.info("****************")
        # logger.info(planning_response_action)
        dict_to_write = {}
        if mode in ["dom", "d_v", "dom_v_desc", "vision_to_dom", "operator"]:
            dict_to_write['id'] = planning_response_action['element_id']
            dict_to_write['action_type'] = planning_response_action['action']
            dict_to_write['value'] = planning_response_action['action_input']
        elif mode == "vision":
            dict_to_write['action'] = planning_response_action['action']
        dict_to_write['description'] = planning_response_action['description']
        dict_to_write['error_message'] = error_message
        dict_to_write['planning_token_count'] = planning_token_count

        return dict_to_write