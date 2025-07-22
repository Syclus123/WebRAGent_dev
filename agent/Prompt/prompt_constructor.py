from ..Utils.utils import is_valid_base64
import json5
import base64
import json

from .vision_to_dom_prompts import VisionToDomPrompts
from .dom_vision_disc_prompts import DomVisionDiscPrompts
from .base_prompts import BasePrompts
from .dom_vision_prompts import DomVisionPrompts
from .vision_prompts import VisionPrompts
from jinja2 import Template
from typing import Union, List, Dict, Any, Optional

from agent.Memory.short_memory.history import HistoryMemory
from agent.Memory.retriever import TestOnlyRetriever

class BasePromptConstructor:
    def __init__(self):
        pass


# Build a prompt for planning based on the DOM tree
class PlanningPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.planning_prompt_system
        self.prompt_user = BasePrompts.planning_prompt_user

    def construct(
            self,
            user_request: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = ""
    ) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
        if len(previous_trace) > 0:
            self.prompt_user += HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            if status_description != "":
                self.prompt_user += \
                    f"Task completion description is {status_description}"
            if feedback != "":
                self.prompt_user += f"Here are some other things you need to know:\n {feedback}\n"
            self.prompt_user += f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"
        messages = [{"role": "system", "content": self.prompt_system}, {
            "role": "user", "content": self.prompt_user}]
        return messages

    # Previous thought, action and reflection are converted to formatted strings
    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i["reflection"]}\";\n'
        str_output += "]"
        return str_output


class VisionDisc2PromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = DomVisionDiscPrompts.dom_vision_disc_prompt_system2
        self.prompt_user = DomVisionDiscPrompts.dom_vision_disc_planning_prompt_user

    def construct(
            self,
            user_request: str,
            base64_image: str
    ) -> list:
        rendered_prompt = Template(self.prompt_user).render(
            user_request=user_request)
        prompt_elements = [{"type": "text", "text": rendered_prompt},
                           {"type": "text", "text": "current web page screenshot is:"},
                           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]

        # Construct the final message payload
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        return messages


class VisionDisc1PromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = DomVisionDiscPrompts.dom_vision_disc_prompt_system1

    def construct(
            self,
            base64_image: str
    ) -> list:
        prompt_elements = [{"type": "text", "text": "current web page screenshot is:"},
                           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]

        # Construct the final message payload
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        return messages


class ObservationVisionDiscPromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = DomVisionDiscPrompts.dom_vision_disc_planning_prompt_system
        self.prompt_user = DomVisionDiscPrompts.dom_vision_disc_planning_prompt_user

    def construct(
            self,
            user_request: str,
            previous_trace: str,
            observation: str,
            feedback: str = "",
            status_description: str = "",
            vision_disc_response: str = ""
    ) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
        if len(previous_trace) > 0:
            self.prompt_user += HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            # if status_description != "":
            #     self.prompt_user += \
            #         f"Task completion description is {status_description}"
            if feedback != "":
                self.prompt_user += f"An invalid action description is below:\n {feedback}\n"
            self.prompt_user += f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"
            if vision_disc_response:
                self.prompt_user += "\n\nHere is a visual analysis of the webpage's screenshot:\n" + \
                    vision_disc_response
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": self.prompt_user}]
        return messages

    # Convert previous thought and action into formatted string
    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}\";\n'
        str_output += "]"
        return str_output


class ObservationVisionActPromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = VisionToDomPrompts.vision_act_planning_prompt_system
        self.prompt_user = VisionToDomPrompts.vision_act_planning_prompt_user

    def construct(
            self,
            user_request: str,
            previous_trace: str,
            observation_vision: str,
            feedback: str = "",
            status_description: str = ""
    ) -> list:
        rendered_prompt = Template(self.prompt_user).render(
            user_request=user_request)
        prompt_elements = [{"type": "text", "text": rendered_prompt}]
        if len(previous_trace) > 0:
            # history_memory = HistoryMemory(previous_trace=previous_trace)
            # trace_prompt = history_memory.construct_previous_trace_prompt()
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            prompt_elements.append({"type": "text", "text": trace_prompt})
            # if status_description != "":
            #     prompt_elements.append({"type": "text", "text": f"Task completion description is {status_description}"})
            if feedback != "":
                prompt_elements.append(
                    {"type": "text", "text": f"An invalid action description is below:\n {feedback}\n"})
            # prompt_elements.append({"type": "text", "text": f"The current webpage's URL is {url}"})
            if observation_vision:
                prompt_elements.append(
                    {"type": "text", "text": "The current webpage's screenshot is:"})
                prompt_elements.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{observation_vision}"}})
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        # print(prompt_elements)
        print("messages finished!\n")
        return messages


class VisionToDomPromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = VisionToDomPrompts.vision_to_dom_planning_prompt_system
        self.prompt_user = ""  # VisionToDomPrompts.vision_act_planning_prompt_user

    def construct(
            self,
            # user_request: str,
            target_element: str,
            action_description: str,
            observation: str
    ) -> list:
        # self.prompt_user = Template(self.prompt_user).render(user_request=user_request)
        self.prompt_user += f"Target Element Description: {target_element}\n"
        if action_description:
            self.prompt_user += f"Action Description: {action_description}\n"
        self.prompt_user += "\nHere is the accessibility tree that you should refer to for this task:\n" + observation
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": self.prompt_user}]
        return messages


class D_VObservationPromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = DomVisionPrompts.d_v_planning_prompt_system
        self.prompt_user = DomVisionPrompts.d_v_planning_prompt_user

    def construct(
            self,
            user_request: str,
            previous_trace: str,
            observation: str,
            observation_VforD: str,
            feedback: str = "",
            status_description: str = ""
    ) -> list:
        is_valid, message = is_valid_base64(
            observation_VforD)
        print("prompt_constructor.py D_VObservationPromptConstructor:", message, "\n")
        rendered_prompt = Template(self.prompt_user).render(
            user_request=user_request)
        prompt_elements = [{"type": "text", "text": rendered_prompt}]
        if len(previous_trace) > 0:
            # history_memory = HistoryMemory(previous_trace=previous_trace)
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            # trace_prompt = history_memory.construct_previous_trace_prompt()
            prompt_elements.append({"type": "text", "text": trace_prompt})
            # if status_description != "":
            #     prompt_elements.append({"type": "text", "text": f"Task completion description is {status_description}"})
            if feedback != "":
                prompt_elements.append(
                    {"type": "text", "text": f"There an invalid action description is below:\n {feedback}\n"})
            prompt_elements.append(
                {"type": "text", "text": f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"})
            prompt_elements.append(
                {"type": "text", "text": "current screenshot is:"})
            print("len of prompt_elements before observation_VforD:",
                  len(prompt_elements))
            prompt_elements_str = json5.dumps(prompt_elements)
            print("len of prompt_elements_str before observation_VforD:", len(
                prompt_elements_str)) # This will print the length of prompt_elements converted into JSON string
            print("len of about gpt token of prompt_elements_str before observation_VforD:", len(
                prompt_elements_str) / 5.42, "\n")
            prompt_elements.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{observation_VforD}"}})
        # Construct the final message payload
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        # print(prompt_elements)
        print("messages finished!\n")
        return messages

    # Convert previous thought and action into formatted string
    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}\";\n'
        str_output += "]"
        return str_output


class VisionObservationPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = VisionPrompts.vision_planning_prompt_system 
        self.prompt_user = VisionPrompts.vision_prompt_user

    def construct(self, user_request: str, previous_trace: str, base64_image: str) -> list:
        rendered_prompt = Template(self.prompt_user).render(
            user_request=user_request)
        prompt_elements = [{"type": "text", "text": rendered_prompt}]

        if len(previous_trace) > 0:
            history_memory = HistoryMemory(previous_trace=[previous_trace])
            trace_prompt = history_memory.construct_previous_trace_prompt()
            prompt_elements.append({"type": "text", "text": trace_prompt})

            prompt_elements.append(
                {"type": "text", "text": "The current observation is:"})
            prompt_elements.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        return messages

    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}\";\n'
        str_output += "]"
        return str_output


class RewardPromptConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = BasePrompts.global_reward_prompt_system
        self.prompt_user = BasePrompts.global_reward_prompt_user

    def construct(
            self,
            ground_truth_mode: str,
            global_reward_mode: str,
            user_request: str,
            stringfy_thought_and_action_output: str,
            observation: str,
            current_info=None,
            instruction: str = ""
    ) -> list:
        if ground_truth_mode:
            self.prompt_system = BasePrompts.global_reward_with_GroundTruth_prompt_system
        rendered_prompt = Template(self.prompt_user).render(
            user_request=user_request, stringfy_thought_and_action_output=stringfy_thought_and_action_output)
        prompt_elements = [{"type": "text", "text": rendered_prompt}]
        if 'current_url' in current_info:
            current_url = current_info.get('current_url', 'not available')
            prompt_elements.append(
                {"type": "text", "text": f"The current url is {current_url}"})
        prompt_elements.append(
            {"type": "text", "text": f"Here is the current accessibility tree that you should refer to:\n{observation}"})
        if "vision" in global_reward_mode:
            if "vision_reward" in current_info and current_info['vision_reward']:
                prompt_elements.append(
                    {"type": "text", "text": "The current screenshot is:"})
                prompt_elements.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_info['vision_reward']}"}})
            else:
                prompt_elements.append(
                    {"type": "text", "text": "The current screenshot is not available."})
                print("The current screenshot for vision reward is not available.")
        if ground_truth_mode:
            prompt_elements.append(
                {"type": "text", "text": f"Here is the Reference Guide for the target task:\n\n{instruction}"})
        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        return messages


# Construct prompt for textual reward
class CurrentRewardPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.current_reward_prompt_system
        self.prompt_user = BasePrompts.current_reward_prompt_user

    def construct(
            self,
            user_request: str,
            stringfy_previous_trace_output: str,
            stringfy_current_trace_output: str,
            observation: str
    ) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request, stringfy_previous_trace_output=stringfy_previous_trace_output,
            stringfy_current_trace_output=stringfy_current_trace_output)
        self.prompt_user += f"\nHere is the accessibility tree that you should refer to:\n{observation}"
        messages = [{"role": "system", "content": self.prompt_system}, {
            "role": "user", "content": self.prompt_user}]
        return messages


# Construct prompt for vision reward
class VisionRewardPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = DomVisionPrompts.current_d_vision_reward_prompt_system
        self.prompt_user = DomVisionPrompts.current_d_vision_reward_prompt_user

    def construct(
            self,
            user_request: str,
            stringfy_previous_trace_output: str,
            stringfy_current_trace_output: str,
            observation: str,
            observation_VforD: str
    ) -> list:
        if not is_valid_base64(observation_VforD):
            print("The observation_VforD provided is not a valid Base64 encoding")

        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request, stringfy_previous_trace_output=stringfy_previous_trace_output,
            stringfy_current_trace_output=stringfy_current_trace_output)
        self.prompt_user += f"the key information of current web page is: {observation}"
        prompt_elements = [{"type": "text", "text": self.prompt_user}]

        prompt_elements.append(
            {"type": "text", "text": "the screenshot of current web page is :"})
        prompt_elements.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{observation_VforD}"}})

        messages = [{"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": prompt_elements}]
        return messages


# Build a prompt to determine whether the element is a search box (if so, the front end needs to add an additional return operation)
class JudgeSearchbarPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.judge_searchbar_prompt_system
        self.prompt_user = BasePrompts.judge_searchbar_prompt_user

    # Build a prompt to determine whether it is a search box, and output a format that can be parsed by openai
    # TODO decoded_result
    def construct(self, input_element, planning_response_action) -> list:
        self.prompt_user = Template(self.prompt_user).render(input_element=str(
            input_element), element_id=planning_response_action['element_id'],
            action_input=planning_response_action['action_input'])
        messages = [{"role": "system", "content": self.prompt_system}, {
            "role": "user", "content": self.prompt_user}]
        return messages


class SemanticMatchPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.semantic_match_prompt_system
        self.prompt_user = BasePrompts.semantic_match_prompt_user

    def construct(self, input_answer, semantic_method) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            semantic_method=semantic_method, input_answer=input_answer)
        messages = [{"role": "system", "content": self.prompt_system}, {
            "role": "user", "content": self.prompt_user}]
        return messages

class ExampleParser:
    """Parser for retrieved examples to format them properly for the prompt."""
    
    @staticmethod
    def parse_action_space(action_space_text: str) -> str:
        """Parse and format the action space description."""
        # Extract action space items and format them
        action_items = []
        for line in action_space_text.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                action_items.append(line.strip())
        return '\n'.join(action_items)
    
    @staticmethod
    def parse_trajectory(trajectory_text: str) -> List[Dict[str, Any]]:
        """Parse the trajectory text into structured steps."""
        steps = []
        current_step = {}
        
        for line in trajectory_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Observation'):
                if current_step:
                    steps.append(current_step)
                current_step = {'observation': line}
            elif line.startswith('Action'):
                try:
                    action_data = json5.loads(line.split(':', 1)[1].strip())
                    current_step['action'] = action_data
                except:
                    current_step['action'] = line
                    
        if current_step:
            steps.append(current_step)
            
        return steps
    
    @staticmethod
    def format_example(task: str, trajectory_text: str) -> str:
        """Format a complete example with task and trajectory."""
        # Extract action space and trajectory
        parts = trajectory_text.split('\n\n')
        action_space = ExampleParser.parse_action_space(parts[0])
        trajectory = ExampleParser.parse_trajectory('\n'.join(parts[1:]))
        
        # Format the example
        formatted = f"Task: {task}\n\n"
        formatted += "Available Actions:\n"
        formatted += action_space + "\n\n"
        formatted += "Example Trajectory:\n"
        
        for step in trajectory:
            formatted += f"Observation: {step['observation']}\n"
            if isinstance(step['action'], dict):
                formatted += f"Action: {json5.dumps(step['action'])}\n"
            else:
                formatted += f"Action: {step['action']}\n"
            formatted += "\n"
            
        return formatted

# Build a prompt for planning based on the DOM tree and retrioeval pool
class PlanningPromptRetrievalConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.planning_prompt_system
        self.prompt_user = BasePrompts.planning_prompt_user
    def construct(
            self,
            user_request: str,
            rag_path: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
    ) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
        self.prompt_user += "## Example Tasks ##\n"
        
        # Setup retrieval paths
        retrieval_path = {
            'collection_path': f"{rag_path}/collection",
            'qry_embed_path': f"{rag_path}/qry_task_embed.json",
            'cand_embed_path': f"{rag_path}/cand_embed.parquet",
            'cand_id_text_path': f"{rag_path}/cand_id_text.json"
        }
        
        # Retrieve examples
        retriever = TestOnlyRetriever(retrieval_path)
        retrieved_tasks, retrieved_texts, retrieved_image_paths = retriever.retrieve(
            task_name=user_request,
        )
        
        # Log the retrieval results
        from log_retrieved_tasks import log_retrieval
        log_retrieval(
            user_request=user_request,
            retrieved_tasks=retrieved_tasks,
            retrieved_texts=retrieved_texts,
            retrieved_image_paths=retrieved_image_paths,
            log_path="Logs/retrieved_tasks.json"
        )
        
        # Format and add examples
        for idx, (task, text) in enumerate(zip(retrieved_tasks, retrieved_texts), 1):
            formatted_example = ExampleParser.format_example(task, text)
            self.prompt_user += f"\nExample {idx}:\n{formatted_example}\n"
            
        # Add previous trace and other information
        if len(previous_trace) > 0:
            self.prompt_user += HistoryMemory(
                previous_trace=previous_trace, 
                reflection=status_description
            ).construct_previous_trace_prompt()
            
            if status_description:
                self.prompt_user += f"\nTask completion description: {status_description}"
                
            if feedback:
                self.prompt_user += f"\nHere are some other things you need to know:\n{feedback}"
                
            self.prompt_user += f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"

        # Construct final messages
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user}
        ]
        
        return messages

    def stringfy_thought_and_action(self, input_data: Union[str, bytes, List[Dict[str, Any]]]) -> str:
        """Convert thought and action data to formatted string.
        
        Args:
            input_data: Input data to format
            
        Returns:
            Formatted string representation
        """
        if isinstance(input_data, (str, bytes)):
            input_list = json5.loads(input_data, encoding="utf-8")
        else:
            input_list = input_data
            
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i.get("reflection", "")}\";\n'
        str_output += "]"
        return str_output

class PlanningPromptVisionRetrievalConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = BasePrompts.planning_prompt_system
        self.prompt_user = BasePrompts.planning_prompt_user
        self.max_image_dimension = 800  # Increased from 800 to 1200 based on analysis

    def scale_image(self, image_path: str) -> bytes:
        """Scale down image while maintaining aspect ratio to reduce size."""
        from PIL import Image
        import io
        
        # Open and scale image
        with Image.open(image_path) as img:
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            if width > height:
                if width > self.max_image_dimension:
                    new_width = self.max_image_dimension
                    new_height = int(height * (self.max_image_dimension / width))
                else:
                    return img.tobytes()
            else:
                if height > self.max_image_dimension:
                    new_height = self.max_image_dimension
                    new_width = int(width * (self.max_image_dimension / height))
                else:
                    return img.tobytes()
            
            # Scale image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG', optimize=True)
            return img_byte_arr.getvalue()

    def parse_retrieved_text(self, text: str) -> tuple:
        """Parse the retrieved text into action space and trajectory steps."""
        parts = text.split('\n\n')
        action_space = parts[0]
        trajectory_text = '\n'.join(parts[1:])
        
        # Parse trajectory into steps
        steps = []
        current_step = {}
        
        for line in trajectory_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Observation'):
                if current_step:
                    steps.append(current_step)
                current_step = {'observation': line}
            elif line.startswith('Action'):
                try:
                    action_data = json5.loads(line.split(':', 1)[1].strip())
                    current_step['action'] = action_data
                except:
                    current_step['action'] = line
                    
        if current_step:
            steps.append(current_step)
            
        return action_space, steps

    def construct(
            self,
            user_request: str,
            rag_path: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
    ) -> list:
        # Start with the base prompt
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
        
        # Setup retrieval paths
        retrieval_path = {
            'collection_path': f"{rag_path}/collection",
            'qry_embed_path': f"{rag_path}/qry_task_embed.json",
            'cand_embed_path': f"{rag_path}/cand_embed_online_mind2web.parquet",
            'cand_id_text_path': f"{rag_path}/cand_id_text2.json" 
        }
        
        # Retrieve examples
        retriever = TestOnlyRetriever(retrieval_path)
        retrieved_tasks, retrieved_texts, retrieved_image_paths, retrieved_workflows = retriever.retrieve(
            task_name=user_request,
        )
        print(f"retrieved_tasks: {retrieved_tasks}")
        print(f"retrieved_texts: {retrieved_texts}")
        print(f"retrieved_image_paths: {retrieved_image_paths}")
        
        # Log the retrieval results
        from log_retrieved_tasks import log_retrieval
        log_retrieval(
            user_request=user_request,
            retrieved_tasks=retrieved_tasks,
            retrieved_texts=retrieved_texts,
            retrieved_image_paths=retrieved_image_paths,
            log_path="Logs/retrieved_tasks.json"
        )

        # Add retrieved examples with their steps and images
        if retrieved_tasks:
            self.prompt_user += "\n\nHere are some similar examples to help you:\n"
            for task, text, image_paths_json in zip(retrieved_tasks, retrieved_texts, retrieved_image_paths):
                # Parse the retrieved text
                action_space, steps = self.parse_retrieved_text(text)

                # Parse the image paths JSON string
                image_paths = json5.loads(image_paths_json)

                # Add task description and action space
                self.prompt_user += f"\nExample:\nTask: {task}\n\n"
                self.prompt_user += f"{action_space}\n\n"
                
                # Add each step with its corresponding image
                prompt_elements = [{"type": "text", "text": self.prompt_user}]
                for step_idx, (step, image_path) in enumerate(zip(steps, image_paths)):
                    # Add the observation with image reference
                    prompt_elements.append({
                        "type": "text", 
                        "text": f"Observation {step_idx + 1}: <|image_{step_idx + 1}|>\n"
                    })
                    # Add the action
                    prompt_elements.append({
                        "type": "text",
                        "text": f"Action {step_idx + 1}: {json5.dumps(step['action']) if isinstance(step['action'], dict) else step['action']}\n"
                    })
                    # Add the corresponding image
                    full_image_path = f"data/Online-Mind2Web/rag_data/image/{image_path}"
                    with open(full_image_path, 'rb') as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        prompt_elements.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        })
                    # try:
                    #     # Scale image before encoding
                    #     scaled_image_bytes = self.scale_image(full_image_path)
                    #     img_base64 = base64.b64encode(scaled_image_bytes).decode('utf-8')
                    #     prompt_elements.append({
                    #         "type": "image_url",
                    #         "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    #     })
                    # except Exception as e:
                    #     print(f"Error processing image {image_path}: {str(e)}")
                    #     # Fallback to original image if scaling fails
                    #     with open(full_image_path, 'rb') as img_file:
                    #         img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    #         prompt_elements.append({
                    #             "type": "image_url",
                    #             "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    #         })
        
        # Add previous trace if exists
        if len(previous_trace) > 0:
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            prompt_elements.append({"type": "text", "text": trace_prompt})
            
            if status_description:
                prompt_elements.append({"type": "text", "text": f"Task completion description is {status_description}"})
            if feedback:
                prompt_elements.append({"type": "text", "text": f"Here are some other things you need to know:\n {feedback}\n"})
            
            prompt_elements.append({"type": "text", "text": f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"})
        
        # Construct the final message payload
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": prompt_elements}
        ]
        return messages

    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i["reflection"]}\";\n'
        str_output += "]"
        return str_output

class PlanningPromptDescriptionRetrievalConstructor(BasePromptConstructor):
    def __init__(self):
        super().__init__()
        self.prompt_system = BasePrompts.planning_prompt_system
        self.prompt_user = BasePrompts.planning_prompt_user
        self.reference = ""  # Initialize as empty string instead of None

    def construct(
            self,
            user_request: str,
            rag_path: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
    ) -> list:
        # Start with the base prompt
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
       
        # Add previous trace if exists
        if len(previous_trace) > 0:
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            self.prompt_user += trace_prompt
            
            if status_description:
                self.prompt_user += f"\nTask completion description: {status_description}"
            if feedback:
                self.prompt_user += f"\nHere are some other things you need to know:\n{feedback}"
            
            self.prompt_user += f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"
        
        # Set reference if none
        if self.reference == "":
            # Load generated descriptions
            descriptions_path = "data/Online-Mind2Web/generated_steps/generated_task_descriptions.json"
            with open(descriptions_path, 'r') as f:
                generated_descriptions = json.load(f)
            
            print(f"\nLooking for task description matching: {user_request}")
            print(f"Available tasks in descriptions file:")
            for desc in generated_descriptions:
                print(f"- {desc['task_name']}")
            
            # Find matching task description
            task_description = next((desc for desc in generated_descriptions if desc['task_name'].lower() == user_request.lower()), None)
            
            if not task_description:
                print(f"\nWARNING: No matching task description found for: {user_request}")
                print("Please ensure the task name matches exactly with one in the descriptions file.")
            else:
                print(f"\nFound matching task description for: {user_request}")
            
            # Add example if found
            if task_description:
                self.reference += "\n\nHere is a similar example to help you:\n"
                self.reference += f"\nTask: {task_description['task_name']}\n\n"
                self.reference += "Steps:\n"
                
                # Add each step with its descriptions
                for step in task_description['steps']:
                    self.reference += f"Step {step['step_number']}:\n"
                    self.reference += f"Observation: {step['observation_description']}\n"
                    self.reference += f"Action: {step['action_description']}\n\n"
            # Add reference confirmation to end of description
            self.reference += "In the final part of your output's description section, you should state whether the similar task example was helpful and elaborate on how your plan draws from it.\n\n"
        
        # Add reference to user prompt
        self.prompt_user += self.reference

        # Construct the final message payload
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user}
        ]
        return messages

    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i["reflection"]}\";\n'
        str_output += "]"
        return str_output


# OpenAI Operator Prompt Constructors
from .operator_prompts import OperatorPrompts

class OperatorPromptConstructor(BasePromptConstructor):
    """
    Operator-specific prompt constructor for browser automation
    """
    
    def __init__(self):
        super().__init__()
        self.prompt_system = OperatorPrompts.operator_planning_system
        self.prompt_user = OperatorPrompts.operator_planning_user
    
    def construct(
            self,
            user_request: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
            screenshot_base64: str = None
    ) -> list:
        """
        Construct prompt for OpenAI Operator
        
        Args:
            user_request: User's task request
            previous_trace: Previous action history
            observation: Current DOM observation
            feedback: Any feedback or error messages
            status_description: Current task status
            screenshot_base64: Current screenshot in base64 format
            
        Returns:
            List of formatted messages for operator
        """
        # Render user prompt with task request
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request
        )
        
        # Start with text content
        content_parts = [{"type": "text", "text": self.prompt_user}]
        
        # Add previous trace if available
        if len(previous_trace) > 0:
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, 
                reflection=status_description
            ).construct_previous_trace_prompt()
            content_parts.append({"type": "text", "text": trace_prompt})
            
            if status_description:
                content_parts.append({
                    "type": "text", 
                    "text": f"Task completion description: {status_description}"
                })
            
            if feedback:
                content_parts.append({
                    "type": "text", 
                    "text": f"Here are some other things you need to know:\n{feedback}"
                })
        
        # Add DOM observation
        if observation:
            content_parts.append({
                "type": "text", 
                "text": f"Current accessibility tree:\n{observation}"
            })
        
        # Add screenshot if available
        if screenshot_base64:
            content_parts.append({
                "type": "text", 
                "text": "Current webpage screenshot:"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}",
                    "detail": "high"
                }
            })
        
        # Construct final messages
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": content_parts}
        ]
        
        return messages

class OperatorPromptRAGConstructor(BasePromptConstructor):
    """
    Operator prompt constructor with RAG support
    """
    
    def __init__(self):
        super().__init__()
        self.prompt_system = OperatorPrompts.operator_rag_system
        self.prompt_user = OperatorPrompts.operator_planning_user
    
    def construct(
            self,
            user_request: str,
            rag_path: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
            screenshot_base64: str = None
    ) -> list:
        """
        Construct prompt for OpenAI Operator with RAG support
        
        Args:
            user_request: User's task request
            rag_path: Path to RAG data
            previous_trace: Previous action history
            observation: Current DOM observation
            feedback: Any feedback or error messages
            status_description: Current task status
            screenshot_base64: Current screenshot in base64 format
            
        Returns:
            List of formatted messages for operator with RAG examples
        """
        # Render user prompt with task request
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request
        )
        
        # Start with text content
        content_parts = [{"type": "text", "text": self.prompt_user}]
        
        # Add RAG examples
        content_parts.append({"type": "text", "text": "## Similar Task Examples ##"})
        
        # Setup retrieval paths
        retrieval_path = {
            'collection_path': f"{rag_path}/collection",
            'qry_embed_path': f"{rag_path}/qry_task_embed.json",
            'cand_embed_path': f"{rag_path}/cand_embed_online_mind2web.parquet",
            'cand_id_text_path': f"{rag_path}/cand_id_text2.json"
        }
        
        try:
            # Retrieve examples
            retriever = TestOnlyRetriever(retrieval_path)
            retrieved_tasks, retrieved_texts, retrieved_image_paths, retrieved_workflows = retriever.retrieve(
                task_name=user_request,
            )
            
            # Log retrieval results
            from log_retrieved_tasks import log_retrieval
            log_retrieval(
                user_request=user_request,
                retrieved_tasks=retrieved_tasks,
                retrieved_texts=retrieved_texts,
                retrieved_image_paths=retrieved_image_paths,
                log_path="Logs/retrieved_tasks.json"
            )
            
            # Add retrieved examples
            if retrieved_tasks:
                content_parts.append({
                    "type": "text", 
                    "text": "\nHere are some similar examples to help guide your actions:"
                })
                
                for task, text, image_paths_json in zip(retrieved_tasks, retrieved_texts, retrieved_image_paths):
                    # Parse the retrieved text
                    action_space, steps = self.parse_retrieved_text(text)
                    
                    # Parse image paths
                    image_paths = json5.loads(image_paths_json)
                    
                    # Add example description
                    content_parts.append({
                        "type": "text", 
                        "text": f"\nExample Task: {task}\n{action_space}\n"
                    })
                    
                    # Add steps with images
                    for step_idx, (step, image_path) in enumerate(zip(steps, image_paths)):
                        content_parts.append({
                            "type": "text", 
                            "text": f"Step {step_idx + 1}: {json5.dumps(step['action']) if isinstance(step['action'], dict) else step['action']}"
                        })
                        
                        # Add corresponding screenshot
                        full_image_path = f"data/Online-Mind2Web/rag_data/image/{image_path}"
                        try:
                            with open(full_image_path, 'rb') as img_file:
                                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                })
                        except Exception as e:
                            print(f"Error loading example image {image_path}: {e}")
                            
        except Exception as e:
            print(f"Error retrieving RAG examples: {e}")
            content_parts.append({
                "type": "text", 
                "text": "No similar examples available at this time."
            })
        
        # Add current task context
        content_parts.append({
            "type": "text", 
            "text": "\n## Current Task Context ##"
        })
        
        # Add previous trace if available
        if len(previous_trace) > 0:
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, 
                reflection=status_description
            ).construct_previous_trace_prompt()
            content_parts.append({"type": "text", "text": trace_prompt})
            
            if status_description:
                content_parts.append({
                    "type": "text", 
                    "text": f"Task completion description: {status_description}"
                })
            
            if feedback:
                content_parts.append({
                    "type": "text", 
                    "text": f"Here are some other things you need to know:\n{feedback}"
                })
        
        # Add DOM observation
        if observation:
            content_parts.append({
                "type": "text", 
                "text": f"Current accessibility tree:\n{observation}"
            })
        
        # Add current screenshot
        if screenshot_base64:
            content_parts.append({
                "type": "text", 
                "text": "Current webpage screenshot:"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}",
                    "detail": "high"
                }
            })
        
        # Construct final messages
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": content_parts}
        ]
        
        return messages
    
    def parse_retrieved_text(self, text: str) -> tuple:
        """
        Parse retrieved text into action space and steps
        """
        try:
            # Simple parsing - adapt based on your data format
            lines = text.split('\n')
            action_space = ""
            steps = []
            
            current_step = {}
            for line in lines:
                if line.startswith('Action:'):
                    if current_step:
                        steps.append(current_step)
                    current_step = {'action': line.replace('Action:', '').strip()}
                elif line.startswith('Thought:'):
                    current_step['thought'] = line.replace('Thought:', '').strip()
                elif line.startswith('Action Space:'):
                    action_space = line.replace('Action Space:', '').strip()
            
            if current_step:
                steps.append(current_step)
            
            return action_space, steps
            
        except Exception as e:
            print(f"Error parsing retrieved text: {e}")
            return "", []

    def stringfy_thought_and_action(self, input_list: list) -> str:
        """Convert thought and action data to formatted string"""
        try:
            if isinstance(input_list, str):
                input_list = json5.loads(input_list, encoding="utf-8")
            
            str_output = "["
            for idx, i in enumerate(input_list):
                str_output += f'Step{idx + 1}:"Thought: {i.get("thought", "")}, Action: {i.get("action", "")}, Reflection: {i.get("reflection", "")}";\n'
            str_output += "]"
            return str_output
            
        except Exception as e:
            print(f"Error stringifying thought and action: {e}")
            return str(input_list)


class OperatorPromptDescriptionRetrievalConstructor(BasePromptConstructor):
    """
    Operator prompt constructor with description-based RAG support
    使用generated_task_descriptions.json中的知识内容构建Operator模式的RAG prompt
    """
    
    def __init__(self):
        super().__init__()
        self.prompt_system = OperatorPrompts.operator_rag_system
        self.prompt_user = OperatorPrompts.operator_planning_user
        self.reference = ""
    
    def construct(
            self,
            user_request: str,
            rag_path: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = "",
            screenshot_base64: str = None
    ) -> list:
        """
        构建带有RAG支持的Operator prompt
        
        Args:
            user_request: 用户任务请求
            rag_path: RAG数据路径（未使用，保持接口一致性）
            previous_trace: 之前的操作历史
            observation: 当前DOM观察（Operator模式中不使用）
            feedback: 反馈或错误消息
            status_description: 当前任务状态
            screenshot_base64: 当前截图的base64编码
            
        Returns:
            格式化的消息列表
        """
        # User prompt
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request
        )
        
        # content parts
        content_parts = [{"type": "input_text", "text": self.prompt_user}]
        
        # task description
        if self.reference == "":
            self.reference = self._load_task_description(user_request)
        
        if self.reference:
            content_parts.append({
                "type": "input_text", 
                "text": "## Similar Task Example ##\n" + self.reference
            })
        
        # 添加当前任务上下文
        content_parts.append({
            "type": "input_text", 
            "text": "\n## Current Task Context ##"
        })
        
        # 添加之前的操作历史
        if len(previous_trace) > 0:
            from agent.Memory.short_memory.history import HistoryMemory
            trace_prompt = HistoryMemory(
                previous_trace=previous_trace, 
                reflection=status_description
            ).construct_previous_trace_prompt()
            content_parts.append({
                "type": "input_text", 
                "text": f"\n**Previous Actions:**\n{trace_prompt}"
            })
            
            if status_description:
                content_parts.append({
                    "type": "input_text", 
                    "text": f"\n**Task Status:** {status_description}"
                })
            
            if feedback:
                content_parts.append({
                    "type": "input_text", 
                    "text": f"\n**Feedback:** {feedback}"
                })
        
        # 添加当前截图
        if screenshot_base64:
            content_parts.append({
                "type": "input_text", 
                "text": "\n**Current Webpage Screenshot:**"
            })
            content_parts.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot_base64}"
            })
        
        # 添加任务指导
        content_parts.append({
            "type": "input_text", 
            "text": """\n## Instructions:
1. Analyze the current screenshot carefully
2. Review the similar task example to understand successful patterns
3. Identify the next logical action to progress toward the goal
4. Provide precise coordinates for any click/drag actions
5. Explain your reasoning and how the example influenced your decision
6. If the task appears complete, indicate this clearly

Please analyze the current state and provide your next action."""
        })
        
        # 构建最终消息
        messages = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": content_parts}
        ]
        
        return messages
    
    def _load_task_description(self, user_request: str) -> str:
        """
        从generated_task_descriptions.json加载任务描述
        
        Args:
            user_request: 用户任务请求
            
        Returns:
            格式化的任务描述字符串
        """
        try:
            # 加载生成的任务描述
            descriptions_path = "data/Online-Mind2Web/generated_steps/generated_task_descriptions.json"
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                generated_descriptions = json.load(f)
            
            print(f"\n🔍 Looking for task description matching: {user_request}")
            print(f"📚 Available tasks in descriptions file: {len(generated_descriptions)}")
            
            # 查找匹配的任务描述
            task_description = None
            
            # 首先尝试精确匹配
            for desc in generated_descriptions:
                if desc['task_name'].lower() == user_request.lower():
                    task_description = desc
                    break
            
            # 尝试部分匹配
            if not task_description:
                for desc in generated_descriptions:
                    if user_request.lower() in desc['task_name'].lower() or desc['task_name'].lower() in user_request.lower():
                        task_description = desc
                        print(f"📝 Found partial match: {desc['task_name']}")
                        break
            
            if not task_description:
                print(f"⚠️  WARNING: No matching task description found for: {user_request}")
                print("Available task examples:")
                for i, desc in enumerate(generated_descriptions[:5]):  # 显示前5个示例
                    print(f"  - {desc['task_name']}")
                if len(generated_descriptions) > 5:
                    print(f"  ... and {len(generated_descriptions) - 5} more")
                return ""
            
            print(f"✅ Found matching task description: {task_description['task_name']}")
            
            # 构建格式化的示例
            reference = f"\n**Example Task:** {task_description['task_name']}\n"
            reference += f"**Website:** {task_description.get('website', 'N/A')}\n"
            reference += f"**Task Level:** {task_description.get('level', 'N/A')}\n\n"
            reference += "**Step-by-step Example:**\n"
            
            # 添加每个步骤的详细描述
            for step in task_description['steps']:
                reference += f"\n**Step {step['step_number']}:**\n"
                reference += f"- **Observation:** {step['observation_description']}\n"
                reference += f"- **Action:** {step['action_description']}\n"
                
                # 如果有原始操作信息，也包含进来
                if 'original_action' in step:
                    original_action = step['original_action']
                    reference += f"- **Operation Type:** {original_action.get('operation', 'N/A')}\n"
                    if original_action.get('value'):
                        reference += f"- **Value:** {original_action['value']}\n"
                    if original_action.get('target'):
                        target = original_action['target']
                        if isinstance(target, dict) and 'x' in target:
                            reference += f"- **Target Coordinates:** ({target['x']:.3f}, {target['y']:.3f})\n"
                
                reference += "\n"
            
            # 添加学习指导
            reference += "\n**Learning Points:**\n"
            reference += "- Pay attention to the sequence of actions and their reasoning\n"
            reference += "- Notice how observations lead to specific actions\n"
            reference += "- Consider the visual elements and their interactions\n"
            reference += "- Adapt the approach to your current task context\n"
            reference += "- Use similar coordinate-based interactions when appropriate\n\n"
            
            return reference
            
        except Exception as e:
            print(f"❌ Error loading task description: {e}")
            return ""
    
    def stringfy_thought_and_action(self, input_list: list) -> str:
        """将思考和行动数据转换为格式化字符串"""
        try:
            if isinstance(input_list, str):
                input_list = json5.loads(input_list, encoding="utf-8")
            
            str_output = "["
            for idx, i in enumerate(input_list):
                str_output += f'Step{idx + 1}:"Thought: {i.get("thought", "")}, Action: {i.get("action", "")}, Reflection: {i.get("reflection", "")}";\n'
            str_output += "]"
            return str_output
            
        except Exception as e:
            print(f"Error stringifying thought and action: {e}")
            return str(input_list)
