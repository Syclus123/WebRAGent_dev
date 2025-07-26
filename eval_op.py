"""
Single task evaluation scripts are supported
Observation and planning of operator patterns are supported

Usage:
    xvfb-run -a python eval_op.py --observation_mode operator --global_reward_mode dom_reward --global_reward_text_model gpt-4o-mini --planning_text_model computer-use-preview-2025-03-11 --single_task_name "Find discussions of the community and open one with the most replies on Flightaware." --single_task_website "https://www.flightaware.com/" --rag_logging_enabled --rag_log_dir test/exp4/rag_logs
"""

from agent.Environment.html_env.async_env import AsyncHTMLEnvironment
from agent.Environment.html_env.operator_env import OperatorEnvironment
from agent.Environment.html_env.operator_actions import OperatorResponseParser, OperatorActionFactory
from evaluate import *
from agent.Plan import *
from dataclasses import dataclass

import re
import asyncio
import argparse
import logging
import json
import time
import os

# universal tools
from agent.Utils.utils import *
# evaluate tools
from evaluate.evaluate_utils import run_task, read_config, read_file
from agent.Utils.utils import read_json_file
from experiment_results import get_evaluate_result

logger = logging.getLogger(__name__)

from agent.LLM.token_utils import is_model_supported


@dataclass
class ExperimentConfig:
    mode: str
    global_reward_mode: str
    planning_text_model: str
    global_reward_text_model: str
    ground_truth_mode: bool
    single_task_name: str
    config: dict
    ground_truth_data: dict
    write_result_file_path: str
    record_time: str
    file: list
    rag_enabled: bool
    rag_path: str
    screenshot_base_dir: str
    rag_logging_enabled: bool
    rag_log_dir: str
    rag_mode: str

def validate_config(config, observation_mode, global_reward_mode, observation_model, global_reward_model):
    """
    Validate configuration, operator mode is supported
    """
    task_mode = config['basic']['task_mode']
    batch_tasks_file_path = config['files']['batch_tasks_file_path']
    json_model_response = config['model']['json_model_response']
    all_json_models = config['model']['json_models']
    interaction_mode = config['steps']['interaction_mode']

    # operator observation mode
    if observation_mode not in ["dom", "operator", "vision", "dom_v_desc", "vision_to_dom", "d_v"]:
        logger.error(
            f"observation mode '{observation_mode}' is not supported! "
            f"Supported modes: dom, operator, vision, dom_v_desc, vision_to_dom, d_v")
        exit()

    if interaction_mode not in [True, False]:
        logger.error(
            "interaction_mode is not defined! Try to define whether you want to evaluate the agent in an interactive manner.")
        exit()

    # json mode
    if observation_mode == "operator":
        logger.info("Using operator mode - JSON mode validation adjusted for operator models")
        if json_model_response:
            if "operator" in observation_model or "computer-use-preview" in observation_model:
                logger.info("Operator model detected - JSON mode will be handled specially")
            elif observation_model not in all_json_models:
                logger.error("Model does not support JSON mode!")
                exit()
    else:
        # Original validation logic
        if json_model_response and (observation_model not in all_json_models or (
                global_reward_mode != 'no_global_reward' and global_reward_model not in all_json_models)):
            logger.error("Model does not support JSON mode!")
            exit()

    if task_mode == 'batch_tasks' and not os.path.exists(batch_tasks_file_path):
        logger.error("batch_tasks_file_path not exist!")
        exit()


def get_task_range(task_mode, file, raw_data_index):
    if task_mode == "batch_tasks":
        if raw_data_index != -1:
            re_result = re.split(r'\s|,', raw_data_index)
            raw_data_start_index = int(re_result[0])
            raw_data_end_index = int(re_result[-1]) + 1
        else:
            raw_data_start_index = 0
            raw_data_end_index = len(file)
        return range(raw_data_start_index, raw_data_end_index)
    elif task_mode == "single_task":
        return range(0, 1)
    else:
        logger.error("task_mode error!")
        exit()


def log_task_info(task_index, task_name, reference_task_length, reference_evaluate_steps):
    logger.info("*" * 100)
    logger.info(f"task index: {task_index}")
    logger.info(f"task name: {task_name}")
    logger.info(f"task reference length: {reference_task_length}")
    logger.info(f"raw data annotation: {reference_evaluate_steps}")


def generate_result_file_path(config):
    return os.path.join(config["files"]["out_file_path"], "json_result")


def load_ground_truth_data(config, ground_truth_mode):
    if ground_truth_mode:
        ground_truth_file_path = config['files']['ground_truth_file_path']
        if not os.path.exists(ground_truth_file_path):
            logger.error("ground_truth_file_path not exist!")
            exit()
        return read_json_file(ground_truth_file_path)
    return None


def create_html_environment(mode, screenshot_dir="screenshots_operator"):
    """
    创建HTML环境，针对operator模式优化
    """
    if mode == "operator":
        # 使用专门的OperatorEnvironment
        return OperatorEnvironment(
            headless=True,  # 确保在服务器环境中使用无头模式
            slow_mo=50,     # 修复：减少延迟时间，提高响应速度
            viewport_width=1280,
            viewport_height=720,
            save_trace_enabled=True,
            screenshot_dir=screenshot_dir
        )
    else:
        # 其他模式使用原有配置
        return AsyncHTMLEnvironment(
            mode=mode,
            max_page_length=8192,
            headless=False,
            slow_mo=1000,
            current_viewport_only=False,
            viewport_size={"width": 1080, "height": 720},
            save_trace_enabled=True,
            sleep_after_execution=0.0,
            locale="en-US",
            use_vimium_effect=True
        )


async def run_operator_task(env, task_name, task_uuid, website, config, 
                          planning_text_model, record_time, write_result_file_path,
                          reference_task_length, reference_evaluate_steps, 
                          screenshot_params, rag_enabled, rag_path, global_reward_mode="no_global_reward", 
                          global_reward_text_model="gpt-4o-mini", ground_truth_mode=False, ground_truth_data=None,
                          rag_logging_enabled=False, rag_log_dir=None, rag_mode="description"):
    """
    运行operator任务 (支持DOM reward和智能停止)
    """
    logger.info(f"🚀 Starting operator task: {task_name}")
    logger.info(f"📱 Model: {planning_text_model}")
    logger.info(f"🌐 Website: {website}")
    logger.info(f"🏆 Reward mode: {global_reward_mode}")
    logger.info(f"🧠 RAG mode: {rag_mode}")
    logger.info(f"📝 RAG logging: {'Enabled' if rag_logging_enabled else 'Disabled'}")
    
    # RAG Logger
    rag_logger = None
    if rag_logging_enabled:
        if rag_mode == "vision":
            from agent.Utils.rag_logger import VisionRAGLogger
            rag_logger = VisionRAGLogger(rag_log_dir=rag_log_dir)
            actual_rag_dir = rag_logger.vision_rag_dir
            logger.info(f"📂 Vision RAG logs will be saved to: {actual_rag_dir}")
        else:
            from agent.Utils.rag_logger import RAGLogger
            rag_logger = RAGLogger(rag_log_dir=rag_log_dir)
            actual_rag_dir = rag_logger.rag_dir
            logger.info(f"📂 RAG logs will be saved to: {actual_rag_dir}")
    
    # 启动环境
    await env.start()
    
    try:
        # 网络检查
        logger.info("🔍 Performing network health check...")
        network_healthy = await env.check_network_health(website)
        if not network_healthy:
            logger.warning("⚠️  Network health check failed, but proceeding with caution...")
        
        await env.navigate_to(website)
        
        # 初始化截图变量
        # current_screenshot = await env.take_screenshot("initial.png")
        current_screenshot = ""
        
        from agent.LLM.llm_instance import create_llm_instance
        
        operator_model = create_llm_instance(
            model=planning_text_model,
            json_mode=False
        )
        
        # OperatorMode
        from agent.Plan.planning import OperatorMode
        operator_mode = OperatorMode(text_model=operator_model)
        
        # 任务执行循环(支持动态步数)
        max_steps = config.get('steps', {}).get('operator_max_steps', 50)  # 增加最大步数限制
        logger.info(f"📊 Using dynamic step limit with maximum: {max_steps}")
        
        step_count = 0
        previous_trace = [] 
        feedback = ""
        status_description = f"Starting task: {task_name}"
        
        task_trace = []
        
        # 添加状态跟踪，避免重复无效操作
        consecutive_failed_scrolls = 0
        last_action_type = None
        
        # DOM reward
        task_finished = False
        task_global_status = ""
        total_reward_score = 0
        consecutive_low_scores = 0  # 连续低分计数
        
        # 获取初始DOM观察
        if hasattr(env, 'get_obs'):
            # 对于operator环境，需要模拟DOM观察
            try:
                # 获取页面标题和URL作为基本观察信息
                page_title = await env.page.title()
                page_url = env.page.url
                observation = f"current web tab name is '{page_title}'\nURL: {page_url}"
            except Exception:
                observation = "Page observation not available"
        else:
            observation = "Operator mode - visual observation only"
        
        logger.info(f"🔄 Starting operator task loop with reward-based stopping...")
        
        while step_count < max_steps:
            logger.info(f"📸 Step {step_count + 1} (max: {max_steps})")
            
            # 每个步骤都保存截图（移除去重机制）
            screenshot_filename = f"step_{step_count:03d}.png"
            
            # 确保页面准备好进行截图
            await ensure_page_ready_for_screenshot(env)
            
            # 获取当前截图
            current_screenshot = await env.take_screenshot(screenshot_filename)
            logger.info(f"📷 Screenshot taken: {screenshot_filename}")
            
            # DOM Reward评估 (在planning之前进行，基于previous trace)
            step_reward = {}
            reward_token_count = [0, 0]
            
            if global_reward_mode != 'no_global_reward' and len(previous_trace) > 0:
                logger.info("🏆 Evaluating DOM reward...")
                try:
                    # 更新观察信息
                    try:
                        page_title = await env.page.title()
                        page_url = env.page.url
                        observation = f"current web tab name is '{page_title}'\nURL: {page_url}"
                    except Exception:
                        observation = "Page observation not available"
                    
                    # 当前信息
                    current_info = {"URL": env.page.url}
                    if "vision" in global_reward_mode:
                        current_info["vision_reward"] = current_screenshot
                    
                    # 调用GlobalReward评估
                    from agent.Reward.global_reward import GlobalReward
                    step_reward, reward_description, reward_token_count = await GlobalReward.evaluate(
                        config=config,
                        model_name=global_reward_text_model,
                        user_request=task_name,
                        previous_trace=previous_trace,
                        observation=observation,
                        current_info=current_info,
                        task_name_id=task_uuid,
                        global_reward_mode=global_reward_mode,
                        ground_truth_mode=ground_truth_mode,
                        ground_truth_data=ground_truth_data,
                    )
                    
                    if step_reward:
                        reward_score = int(step_reward.get("score", 0))
                        reward_status = step_reward.get("status", "doing")
                        total_reward_score += reward_score
                        
                        logger.info(f"🏆 Reward Score: {reward_score}/10")
                        logger.info(f"📊 Status: {reward_status}")
                        logger.info(f"💭 Reason: {step_reward.get('reason', 'No reason provided')}")
                        
                        # 智能停止判断
                        if reward_status == "finished" or reward_score == 10:
                            logger.info("🎯 Task completed based on reward evaluation!")
                            task_global_status = "finished"
                            task_finished = True
                            break
                        elif reward_status == "loop" or reward_score == 1:
                            consecutive_low_scores += 1
                            logger.warning(f"⚠️  Low score detected ({consecutive_low_scores}/3)")
                            if consecutive_low_scores >= 5:
                                logger.warning("🔄 Stopping due to consecutive low scores - task may be stuck")
                                task_global_status = "loop"
                                break
                        elif reward_score <= 3:
                            consecutive_low_scores += 1
                            if consecutive_low_scores >= 8:
                                logger.warning("🔄 Stopping due to persistent low performance")
                                task_global_status = "low_performance"
                                break
                        else:
                            consecutive_low_scores = 0  # 重置低分计数
                        
                        # 更新状态描述
                        status_description = reward_description or f"Step {step_count + 1} - Score: {reward_score}"
                        
                except Exception as e:
                    logger.error(f"❌ Error in reward evaluation: {e}")
                    step_reward = {}
            
            # 执行planning
            try:
                # 将previous_trace转换为字符串格式用于operator
                previous_trace_str = ""
                for i, trace in enumerate(previous_trace):
                    previous_trace_str += f"Step {i + 1}: {trace.get('thought', '')} -> {trace.get('action', '')}\n"
                
                planning_response, error_message, planning_response_thought, planning_response_action, planning_token_count, rag_data = await operator_mode.execute(
                    status_description=status_description,
                    user_request=task_name,
                    rag_enabled=rag_enabled,
                    rag_path=rag_path,
                    previous_trace=previous_trace_str,
                    observation="",  # operator模式不需要DOM观察
                    feedback=feedback,
                    observation_VforD=current_screenshot,
                    rag_mode=rag_mode
                )
                
                if error_message:
                    logger.error(f"Planning error: {error_message}")
                    break
                
                logger.info(f"Step Thought: {planning_response_thought}")
                logger.info(f"Step Action: {planning_response_action}")
                
                # RAG logger
                if rag_logging_enabled and rag_logger and rag_data:
                    try:
                        # 添加额外的步骤信息
                        rag_data.update({
                            "step_idx": step_count,
                            "task_name": task_name,
                            "website": website,
                            "model": planning_text_model,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "planning_response": planning_response,
                            "planning_thought": planning_response_thought,
                            "planning_action": planning_response_action,
                            "feedback": feedback,
                            "status_description": status_description,
                            "token_count": planning_token_count,
                            "screenshot_filename": screenshot_filename
                        })
                        
                        rag_file_path = rag_logger.log_rag_step(task_uuid, step_count, rag_data)
                        logger.info(f"📝 RAG information logged to: {rag_file_path}")
                        
                    except Exception as rag_error:
                        logger.warning(f"⚠️  Failed to log RAG information: {rag_error}")
                
                # 检测是否是重复的无效操作
                current_action_type = planning_response_action.get("action", "")
                
                # 改进的重复操作检测
                if is_repetitive_action(current_action_type, last_action_type, consecutive_failed_scrolls):
                    consecutive_failed_scrolls += 1
                    if consecutive_failed_scrolls >= 3:
                        logger.warning("⚠️  Detected repeated ineffective actions, trying alternative strategy")
                        feedback = "Previous actions seem ineffective. Try a different approach, look for alternative UI elements, or consider the task might be completed."
                        consecutive_failed_scrolls = 0
                        
                        # 尝试智能恢复策略
                        alternative_action = await suggest_alternative_action(env, current_action_type, task_name)
                        if alternative_action:
                            planning_response_action = alternative_action
                            logger.info(f"🔄 Switching to alternative action: {alternative_action}")
                else:
                    consecutive_failed_scrolls = 0
                
                last_action_type = current_action_type
                
                # 记录trace (包含reward信息)
                trace_entry = {
                    "step": step_count,
                    "thought": planning_response_thought,
                    "action": planning_response_action,
                    "screenshot_taken": True, # 每个步骤都截图
                    "screenshot": screenshot_filename,
                    "reward": step_reward,  # 添加reward信息
                    "reward_tokens": reward_token_count
                }
                task_trace.append(trace_entry)
                
                # 为下一轮reward评估准备trace数据
                current_trace = {
                    "thought": planning_response_thought,
                    "action": planning_response_action.get("action", ""),
                    "action_input": planning_response_action.get("action_input", ""),
                    "reflection": step_reward.get("description", "") if step_reward else ""
                }
                
                # 执行action
                success = await execute_operator_action(env, planning_response_action)
                
                if not success:
                    logger.error("Action execution failed")
                    feedback = "Action execution failed. Please try a different approach or simpler actions."
                    current_trace["reflection"] += " (Action execution failed)"
                else:
                    feedback = ""
                
                # 添加到previous_trace用于下一轮reward评估
                previous_trace.append(current_trace)
                
                # 检查是否完成
                if planning_response_action.get("action") == "get_final_answer":
                    logger.info("✅ Task completed by final answer!")
                    task_finished = True
                    break
                
                step_count += 1
                
            except Exception as e:
                logger.error(f"Error in planning step: {e}")
                feedback = f"Error in planning: {str(e)}"
                break
        
        # 计算最终状态
        if task_finished:
            final_status = "finished"
        elif task_global_status == "finished":
            final_status = "llm_finished"
        elif task_global_status == "loop":
            final_status = "loop_detected"
        elif task_global_status == "low_performance":
            final_status = "low_performance"
        elif step_count >= max_steps:
            final_status = "step_limit"
        else:
            final_status = "unknown"
        
        result_data = {
            "task_name": task_name,
            "task_uuid": task_uuid,
            "model": planning_text_model,
            "website": website,
            "steps": step_count,
            "max_steps": max_steps,
            "final_status": final_status,
            "task_global_status": task_global_status,
            "total_reward_score": total_reward_score,
            "average_reward_score": total_reward_score / max(1, step_count),
            "reward_mode": global_reward_mode,
            "trace": task_trace,
            "final_state": await env.get_current_state(),
            "completed": task_finished or task_global_status == "finished",
            "record_time": record_time,
            "reward_based_stopping": global_reward_mode != 'no_global_reward'
        }
        
        result_file = os.path.join(write_result_file_path, f"{task_uuid}_{record_time}.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Result saved to: {result_file}")
        logger.info(f"🏆 Final Status: {final_status}")
        logger.info(f"📊 Total Steps: {step_count}")
        logger.info(f"🎯 Total Reward Score: {total_reward_score}")
        if step_count > 0:
            logger.info(f"📈 Average Reward Score: {total_reward_score / step_count:.2f}")
        
    except Exception as navigation_error:
        logger.error(f"❌ Navigation failed: {navigation_error}")
        
        # 尝试恢复策略
        logger.info("🔄 Attempting recovery strategies...")
        
        try:
            # 策略1：尝试导航到备用URL或简单页面
            if "flightaware" in website.lower():
                backup_url = "https://www.google.com"
                logger.info(f"🔄 Trying backup URL: {backup_url}")
                await env.navigate_to(backup_url, max_retries=2)
                
                # 从Google搜索目标站点
                logger.info("🔍 Searching for target site from Google...")
                
            else:
                # 其他站点的备用策略
                logger.info("🔄 Using fallback navigation...")
                await env.navigate_to("about:blank")
                
        except Exception as recovery_error:
            logger.error(f"❌ Recovery also failed: {recovery_error}")
            # 保存错误结果
            result_data = {
                "task_name": task_name,
                "task_uuid": task_uuid,
                "model": planning_text_model,
                "website": website,
                "error": "Navigation failed",
                "error_details": str(navigation_error),
                "recovery_error": str(recovery_error),
                "steps": 0,
                "max_steps": max_steps,
                "final_status": "navigation_error",
                "completed": False,
                "record_time": record_time,
                "reward_mode": global_reward_mode
            }
            
            # 保存错误结果
            result_file = os.path.join(write_result_file_path, f"{task_uuid}_{record_time}_error.json")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.error(f"💾 Error result saved to: {result_file}")
            return  # 退出任务
        
        # 如果恢复成功，继续任务
        logger.info("✅ Recovery successful, continuing with task...")
        
        # 重新初始化截图和模型
        current_screenshot = await env.take_screenshot()
        
        from agent.LLM.llm_instance import create_llm_instance
        
        operator_model = create_llm_instance(
            model=planning_text_model,
            json_mode=False
        )
        
        # 创建OperatorMode实例
        from agent.Plan.planning import OperatorMode
        operator_mode = OperatorMode(text_model=operator_model)
        
    finally:
        # 关闭环境
        await env.close()


async def execute_operator_action(env, action_dict):
    """
    执行operator action (优化版本 - 减少不必要的wait操作)
    """
    action_type = action_dict.get("action", "wait")
    
    try:
        logger.info(f"🔧 Executing action: {action_type}")
        
        # 记录操作前的页面状态
        page_state_before = await get_page_state(env)
        
        if action_type == "operator_click":
            coords = action_dict.get("coordinates", [0, 0])
            logger.info(f"📍 Clicking at coordinates: {coords}")
            action = OperatorActionFactory.create_click_action(coords[0], coords[1])
            await env.execute_operator_actions([action])
            
            # 点击后智能等待页面稳定
            await wait_for_page_stability(env, expected_change=True)
            
        elif action_type == "operator_double_click":
            coords = action_dict.get("coordinates", [0, 0])
            logger.info(f"📍 Double-clicking at coordinates: {coords}")
            action = OperatorActionFactory.create_double_click_action(coords[0], coords[1])
            await env.execute_operator_actions([action])
            
            # 双击后智能等待页面稳定
            await wait_for_page_stability(env, expected_change=True)
            
        elif action_type == "operator_type":
            text = action_dict.get("text", "")
            logger.info(f"⌨️  Typing text: '{text}'")
            action = OperatorActionFactory.create_type_action(text)
            await env.execute_operator_actions([action])
            
            # 文本输入后短暂等待（不需要长时间等待）
            await asyncio.sleep(0.3)
            
        elif action_type == "operator_scroll":
            scroll_x = action_dict.get("scroll_x", 0)
            scroll_y = action_dict.get("scroll_y", 0)
            
            # 修复：将过大的滚动量调整为合理范围
            if abs(scroll_y) > 500:
                scroll_y = 500 if scroll_y > 0 else -500
            if abs(scroll_x) > 500:
                scroll_x = 500 if scroll_x > 0 else -500
                
            logger.info(f"📜 Scrolling: x={scroll_x}, y={scroll_y}")
            action = OperatorActionFactory.create_scroll_action(scroll_x, scroll_y)
            await env.execute_operator_actions([action])
            
            # 滚动后智能等待内容稳定
            await wait_for_scroll_completion(env)
            
        elif action_type == "operator_keypress":
            keys = action_dict.get("keys", [])
            logger.info(f"🔑 Pressing keys: {keys}")
            action = OperatorActionFactory.create_keypress_action(keys)
            await env.execute_operator_actions([action])
            
            # 按键后智能等待（某些按键可能触发页面变化）
            if any(key.lower() in ['enter', 'return', 'tab'] for key in keys):
                await wait_for_page_stability(env, expected_change=True, timeout=3000)
            else:
                await asyncio.sleep(0.2)
            
        elif action_type == "operator_drag":
            path = action_dict.get("path", [[0, 0], [0, 0]])
            logger.info(f"🖱️  Dragging from {path[0]} to {path[-1]}")
            action = OperatorActionFactory.create_drag_action(path)
            await env.execute_operator_actions([action])
            
            # 拖拽后等待页面稳定
            await wait_for_page_stability(env, expected_change=True)
            
        elif action_type == "operator_wait":
            ms = action_dict.get("ms", 1000)
            # 限制等待时间在合理范围内
            ms = min(ms, 5000)  # 最多等待5秒
            logger.info(f"⏳ Waiting for {ms}ms")
            action = OperatorActionFactory.create_wait_action(ms)
            await env.execute_operator_actions([action])
            
        elif action_type == "get_final_answer":
            logger.info(f"🎯 Task completion detected")
            # 任务完成，不需要额外等待
            return True
            
        else:
            # 优化：对于未知操作，不再默认等待，而是尝试解析或跳过
            logger.warning(f"❓ Unknown action type '{action_type}', attempting minimal response")
            
            # 检查是否是有效的操作描述
            if "click" in action_type.lower():
                # 尝试解析为点击操作
                coords = [640, 360]  # 屏幕中心作为默认点击位置
                action = OperatorActionFactory.create_click_action(coords[0], coords[1])
                await env.execute_operator_actions([action])
                await wait_for_page_stability(env, expected_change=True)
            elif "scroll" in action_type.lower():
                # 尝试解析为滚动操作
                action = OperatorActionFactory.create_scroll_action(0, 200)
                await env.execute_operator_actions([action])
                await wait_for_scroll_completion(env)
            else:
                # 真正的未知操作，最小等待
                logger.info(f"⚡ Minimal wait for unknown action")
                await asyncio.sleep(0.5)  # 减少到0.5秒
        
        # 检查页面状态变化
        page_state_after = await get_page_state(env)
        if page_state_changed(page_state_before, page_state_after):
            logger.info(f"📄 Page state changed, ensuring stability...")
            await ensure_page_ready_for_screenshot(env)
        
        logger.info(f"✅ Action '{action_type}' completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error executing operator action {action_type}: {e}")
        return False


async def get_page_state(env):
    """获取页面状态快照"""
    try:
        return {
            "url": env.page.url,
            "title": await env.page.title(),
            "scroll_position": await env.page.evaluate("window.pageYOffset"),
            "timestamp": time.time()
        }
    except Exception:
        return {"timestamp": time.time()}


def page_state_changed(state_before, state_after):
    """检查页面状态是否发生变化"""
    if not state_before or not state_after:
        return True
    
    # 检查URL变化
    if state_before.get("url") != state_after.get("url"):
        return True
    
    # 检查标题变化
    if state_before.get("title") != state_after.get("title"):
        return True
    
    # 检查滚动位置变化（超过50像素认为有显著变化）
    scroll_before = state_before.get("scroll_position", 0)
    scroll_after = state_after.get("scroll_position", 0)
    if abs(scroll_before - scroll_after) > 50:
        return True
    
    return False


async def wait_for_page_stability(env, expected_change=False, timeout=5000):
    """
    智能等待页面稳定
    
    Args:
        env: 环境实例
        expected_change: 是否期望页面发生变化
        timeout: 超时时间（毫秒）
    """
    try:
        start_time = time.time()
        
        # 策略1：等待网络请求完成
        try:
            await env.page.wait_for_load_state("networkidle", timeout=min(3000, timeout))
            logger.info("✅ Network idle achieved")
            return
        except Exception:
            logger.debug("Network idle timeout, trying alternative strategies...")
        
        # 策略2：等待DOM稳定
        stable_count = 0
        last_dom_size = 0
        
        while (time.time() - start_time) * 1000 < timeout:
            try:
                # 检查DOM大小
                dom_size = await env.page.evaluate("document.documentElement.innerHTML.length")
                
                if dom_size == last_dom_size:
                    stable_count += 1
                    if stable_count >= 3:  # 连续3次检查DOM大小相同，认为稳定
                        logger.info("✅ DOM stability achieved")
                        return
                else:
                    stable_count = 0
                    last_dom_size = dom_size
                
                await asyncio.sleep(0.5)
                
            except Exception:
                break
        
        # 策略3：基础等待
        logger.debug("Using fallback wait strategy")
        await asyncio.sleep(1)
        
    except Exception as e:
        logger.warning(f"Page stability check failed: {e}")


async def wait_for_scroll_completion(env):
    """等待滚动操作完成"""
    try:
        # 等待滚动动画完成
        last_scroll = -1
        stable_count = 0
        
        for _ in range(10):  # 最多检查10次
            current_scroll = await env.page.evaluate("window.pageYOffset")
            
            if current_scroll == last_scroll:
                stable_count += 1
                if stable_count >= 2:  # 连续2次位置相同，认为滚动完成
                    logger.info("✅ Scroll completion detected")
                    return
            else:
                stable_count = 0
                last_scroll = current_scroll
            
            await asyncio.sleep(0.2)
        
        logger.info("✅ Scroll timeout reached, assuming completion")
        
    except Exception as e:
        logger.warning(f"Scroll completion check failed: {e}")


async def ensure_page_ready_for_screenshot(env):
    """确保页面已准备好进行截图"""
    try:
        # 等待渲染完成
        await env.page.wait_for_timeout(500)
        
        # 检查是否有加载指示器
        try:
            loading_indicators = await env.page.evaluate("""
                () => {
                    const indicators = [
                        'div[class*="loading"]',
                        'div[class*="spinner"]',
                        'div[class*="loader"]',
                        '.loading',
                        '.spinner',
                        '.loader'
                    ];
                    
                    for (const selector of indicators) {
                        const elements = document.querySelectorAll(selector);
                        for (const el of elements) {
                            const style = window.getComputedStyle(el);
                            if (style.display !== 'none' && style.visibility !== 'hidden') {
                                return true;
                            }
                        }
                    }
                    return false;
                }
            """)
            
            if loading_indicators:
                logger.info("⏳ Waiting for loading indicators to disappear...")
                # 等待加载指示器消失
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    still_loading = await env.page.evaluate("""
                        () => {
                            const indicators = [
                                'div[class*="loading"]',
                                'div[class*="spinner"]', 
                                'div[class*="loader"]',
                                '.loading',
                                '.spinner',
                                '.loader'
                            ];
                            
                            for (const selector of indicators) {
                                const elements = document.querySelectorAll(selector);
                                for (const el of elements) {
                                    const style = window.getComputedStyle(el);
                                    if (style.display !== 'none' && style.visibility !== 'hidden') {
                                        return true;
                                    }
                                }
                            }
                            return false;
                        }
                    """)
                    
                    if not still_loading:
                        logger.info("✅ Loading indicators disappeared")
                        break
            
            logger.info("✅ Page ready for screenshot")
            
        except Exception:
            logger.debug("Loading indicator check failed, proceeding anyway")
            
    except Exception as e:
        logger.warning(f"Page readiness check failed: {e}")


async def get_page_signature(env):
    """
    获取页面签名，用于检测页面是否发生实质性变化
    """
    try:
        signature_data = await env.page.evaluate("""
            () => {
                try {
                    // 获取页面的关键特征
                    const url = window.location.href;
                    const title = document.title;
                    const scrollPosition = window.pageYOffset;
                    const visibleText = document.body ? document.body.innerText.substring(0, 500) : ''; // 前500字符
                    const elementCount = document.querySelectorAll('*').length;
                    
                    // 获取主要内容区域的特征
                    const mainContent = document.querySelector('main, #main, .main, #content, .content, .container');
                    const mainText = mainContent ? mainContent.innerText.substring(0, 200) : '';
                    
                    // 安全的文本哈希函数，避免btoa编码问题
                    function safeHash(text) {
                        if (!text) return '';
                        
                        try {
                            // 简单的字符串哈希函数
                            let hash = 0;
                            for (let i = 0; i < text.length; i++) {
                                const char = text.charCodeAt(i);
                                hash = ((hash << 5) - hash) + char;
                                hash = hash & hash; // 转换为32位整数
                            }
                            return hash.toString(36); // 转换为36进制字符串
                        } catch (e) {
                            console.warn('Text hashing failed:', e);
                            return text.length.toString(); // 备用：返回文本长度
                        }
                    }
                    
                    return {
                        url: url || '',
                        title: title || '',
                        scrollPosition: Math.floor(scrollPosition / 100) * 100, // 量化滚动位置，减少小幅滚动的影响
                        visibleTextHash: safeHash(visibleText), // 使用安全的哈希函数
                        elementCount: elementCount || 0,
                        mainTextHash: safeHash(mainText)
                    };
                } catch (error) {
                    console.error('Page signature generation failed:', error);
                    // 返回最基本的签名数据
                    return {
                        url: window.location.href || '',
                        title: document.title || '',
                        scrollPosition: 0,
                        visibleTextHash: 'error',
                        elementCount: 0,
                        mainTextHash: 'error'
                    };
                }
            }
        """)
        
        # 验证签名数据的完整性
        if not signature_data:
            logger.warning("Page signature data is empty, using fallback")
            signature_data = {
                'url': 'unknown',
                'title': 'unknown', 
                'scrollPosition': 0,
                'visibleTextHash': 'fallback',
                'elementCount': 0,
                'mainTextHash': 'fallback'
            }
        
        # 创建页面签名
        import hashlib
        signature_string = f"{signature_data['url']}|{signature_data['title']}|{signature_data['scrollPosition']}|{signature_data['visibleTextHash']}|{signature_data['elementCount']}|{signature_data['mainTextHash']}"
        signature_hash = hashlib.md5(signature_string.encode('utf-8')).hexdigest()
        
        logger.debug(f"Page signature generated: {signature_hash[:8]}... (URL: {signature_data.get('url', 'unknown')[:50]})")
        
        return signature_hash
        
    except Exception as e:
        logger.warning(f"Failed to get page signature: {e}")
        # 返回基于时间和URL的备用签名
        try:
            current_url = await env.page.url if env.page else "unknown"
            fallback_string = f"{current_url}|{time.time()}"
            import hashlib
            return hashlib.md5(fallback_string.encode('utf-8')).hexdigest()
        except Exception:
            # 最后的备用方案
            return str(time.time())


def is_repetitive_action(current_action, last_action, consecutive_count):
    """
    检测是否是重复的无效操作
    
    Args:
        current_action: 当前操作类型
        last_action: 上一个操作类型
        consecutive_count: 连续相同操作计数
        
    Returns:
        bool: 是否是重复操作
    """
    # 连续相同的操作类型
    if current_action == last_action:
        # 某些操作连续执行可能是正常的（如滚动浏览内容）
        if current_action in ["operator_scroll"]:
            return consecutive_count >= 2  # 滚动操作允许2次
        elif current_action in ["operator_click", "operator_type"]:
            return consecutive_count >= 1  # 点击和输入操作不允许连续
        elif current_action in ["operator_wait"]:
            return consecutive_count >= 0  # wait操作立即被视为重复
        else:
            return consecutive_count >= 1
    
    return False


async def suggest_alternative_action(env, failed_action_type, task_name):
    """
    基于失败的操作类型和任务内容，建议替代操作
    
    Args:
        env: 环境实例
        failed_action_type: 失败的操作类型
        task_name: 任务名称
        
    Returns:
        dict: 建议的替代操作，如果没有建议则返回None
    """
    try:
        page_info = await env.page.evaluate("""
            () => {
                // 检查页面上的可交互元素
                const clickableElements = document.querySelectorAll('button, a, input[type="submit"], input[type="button"], [role="button"]');
                const inputElements = document.querySelectorAll('input[type="text"], input[type="search"], textarea');
                const scrollableElements = document.querySelectorAll('[style*="overflow"], .scroll, .scrollable');
                
                // 检查是否有搜索相关元素
                const searchElements = document.querySelectorAll('input[type="search"], input[placeholder*="search"], input[placeholder*="Search"], .search-input');
                
                // 检查是否有导航元素
                const navElements = document.querySelectorAll('nav, .nav, .navigation, .menu, [role="navigation"]');
                
                return {
                    hasClickableElements: clickableElements.length > 0,
                    hasInputElements: inputElements.length > 0,
                    hasSearchElements: searchElements.length > 0,
                    hasNavElements: navElements.length > 0,
                    clickableCount: clickableElements.length,
                    inputCount: inputElements.length,
                    currentUrl: window.location.href,
                    pageTitle: document.title
                };
            }
        """)
        
        # 基于失败的操作类型和页面状态建议替代方案
        if failed_action_type == "operator_scroll":
            # 滚动失败，尝试其他导航方式
            if page_info["hasNavElements"]:
                return {
                    "action": "operator_click", 
                    "coordinates": [640, 100], # 点击导航区域
                    "action_input": "640,100",
                    "element_id": "nav_alternative"
                }
            elif page_info["hasClickableElements"]:
                return {
                    "action": "operator_click",
                    "coordinates": [640, 300], # 点击页面中部
                    "action_input": "640,300", 
                    "element_id": "click_alternative"
                }
        
        elif failed_action_type == "operator_click":
            # 点击失败，尝试搜索或其他交互
            if page_info["hasSearchElements"] and any(keyword in task_name.lower() for keyword in ["search", "find", "look"]):
                return {
                    "action": "operator_click",
                    "coordinates": [640, 200], # 点击搜索区域
                    "action_input": "640,200",
                    "element_id": "search_alternative"
                }
            else:
                # 尝试按键操作
                return {
                    "action": "operator_keypress", 
                    "keys": ["Tab"],
                    "action_input": "Tab",
                    "element_id": "tab_alternative"
                }
        
        elif failed_action_type == "operator_type":
            # 输入失败，尝试点击输入框或清空后重试
            return {
                "action": "operator_keypress",
                "keys": ["Control", "a"], # 全选
                "action_input": "Control,a",
                "element_id": "select_all_alternative"
            }
        
        # 通用备选方案：按键操作
        return {
            "action": "operator_keypress",
            "keys": ["Escape"], # ESC键可能关闭弹窗或重置状态
            "action_input": "Escape",
            "element_id": "escape_alternative"
        }
        
    except Exception as e:
        logger.warning(f"Failed to suggest alternative action: {e}")
        return None


async def run_experiment(task_range, experiment_config):
    for task_index in task_range:
        task_uuid = None
        if experiment_config.config['basic']['task_mode'] == "batch_tasks":
            task = experiment_config.file[task_index]
            task_name = task.get("confirmed_task", f"Task_{task_index}")
            task_uuid = task.get("task_id", f"task_{task_index}")
            reference_task_length = task.get("reference_length", 0)
            reference_evaluate_steps = task.get("evaluation", [])
            website = task.get("website", "about:blank")
            log_task_info(task_index, task_name,
                          reference_task_length, reference_evaluate_steps)
        elif experiment_config.config['basic']['task_mode'] == "single_task":
            task_name = experiment_config.single_task_name
            reference_task_length = experiment_config.config['steps']['single_task_action_step']
            reference_evaluate_steps = []
            website = experiment_config.config.get('single_task_website', "about:blank")
            task_uuid = f"single_task_{int(time.time())}"
            
            logger.info(f"task_name: {task_name}")
            logger.info(f"website: {website}")
            logger.info(f"mode: {experiment_config.mode}")

        # trajectory parameters
        screenshot_params = {
            "mode": experiment_config.mode,
            "record_time": experiment_config.record_time,
            "task_name": task_name,
            "task_name_id": task_uuid,
            "file_path": experiment_config.screenshot_base_dir or "results_operator"
        }
        
        # trajectory directory
        if experiment_config.screenshot_base_dir:
            base_screenshot_dir = os.path.join(experiment_config.screenshot_base_dir, "img_screenshots")
        else:
            base_screenshot_dir = os.path.join("results_operator", "img_screenshots")
        
        # trajectory directory
        safe_task_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task_name = safe_task_name.replace(' ', '_')[:50]  # long name limit length to avoid path too long
        if not safe_task_name:
            safe_task_name = f"task_{task_uuid}"
        
        task_screenshot_dir = os.path.join(base_screenshot_dir, safe_task_name)
        if not os.path.exists(task_screenshot_dir):
            os.makedirs(task_screenshot_dir, exist_ok=True)
        
        # Operator mode uses special task running logic
        if experiment_config.mode == "operator":
            env = create_html_environment(experiment_config.mode, screenshot_dir=task_screenshot_dir)
            
            logger.info(f"🤖 Using OpenAI Operator mode")
            logger.info(f"📱 Planning model: {experiment_config.planning_text_model}")
            logger.info(f"📸 Screenshot support: Enabled")
            logger.info(f"🧠 RAG support: {'Enabled' if experiment_config.rag_enabled else 'Disabled'}")
            
            await run_operator_task(
                env=env,
                task_name=task_name,
                task_uuid=task_uuid,
                website=website,
                config=experiment_config.config,
                planning_text_model=experiment_config.planning_text_model,
                record_time=experiment_config.record_time,
                write_result_file_path=experiment_config.write_result_file_path,
                reference_task_length=reference_task_length,
                reference_evaluate_steps=reference_evaluate_steps,
                screenshot_params=screenshot_params,
                rag_enabled=experiment_config.rag_enabled,
                rag_path=experiment_config.rag_path,
                global_reward_mode=experiment_config.global_reward_mode,
                global_reward_text_model=experiment_config.global_reward_text_model,
                ground_truth_mode=experiment_config.ground_truth_mode,
                ground_truth_data=experiment_config.ground_truth_data,
                rag_logging_enabled=experiment_config.rag_logging_enabled,
                rag_log_dir=experiment_config.rag_log_dir,
                rag_mode=experiment_config.rag_mode
            )
        else:
            # Other modes use the original logic
            env = create_html_environment(experiment_config.mode, screenshot_dir=task_screenshot_dir)
            
            await run_task(mode=experiment_config.mode,
                           task_mode=experiment_config.config['basic']['task_mode'],
                           task_name=task_name,
                           task_uuid=task_uuid,
                           config=experiment_config.config,
                           write_result_file_path=experiment_config.write_result_file_path,
                           reference_task_length=reference_task_length,
                           evaluate_steps=reference_evaluate_steps,
                           reference_evaluate_steps=reference_evaluate_steps,
                           env=env,
                           global_reward_mode=experiment_config.global_reward_mode,
                           global_reward_text_model=experiment_config.global_reward_text_model,
                           planning_text_model=experiment_config.planning_text_model,
                           ground_truth_mode=experiment_config.ground_truth_mode,
                           ground_truth_data=experiment_config.ground_truth_data,
                           interaction_mode=experiment_config.config['steps']['interaction_mode'],
                           task_index=task_index,
                           record_time=experiment_config.record_time,
                           token_pricing=experiment_config.config['token_pricing'],
                           screenshot_params=screenshot_params,
                           website=website,
                           rag_enabled=experiment_config.rag_enabled,
                           rag_path=experiment_config.rag_path
                           )
            
            await env.close()
            del env
        
    logger.info('\033[31m🎉 All tasks finished!\033[0m')
    logger.info('\033[31m⏸️  Press Enter to exit...\033[0m')


async def main(global_reward_mode="no_global_reward",
               planning_text_model="computer-use-preview-2025-03-11",
               global_reward_text_model="gpt-4o-mini",
               single_task_name="",
               single_task_website="about:blank",
               raw_data_index=-1,
               observation_mode="operator",
               ground_truth_mode=False,
               toml_path=None,
               screenshot_base_dir=None,
               rag_logging_enabled=False,
               rag_log_dir=None,
               rag_mode="description"
               ):
    config = read_config(toml_path)
    config['single_task_website'] = single_task_website
    validate_config(config, observation_mode, global_reward_mode, planning_text_model, global_reward_text_model)

    file = None
    if config['basic']['task_mode'] == "batch_tasks":
        file = read_json_file(config['files']['batch_tasks_file_path'])
        task_range = get_task_range(
            config['basic']['task_mode'], file, raw_data_index)
    elif config['basic']['task_mode'] == "single_task":
        task_range = get_task_range(config['basic']['task_mode'], None, -1)

    record_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    write_result_file_path = generate_result_file_path(config)
    ground_truth_data = load_ground_truth_data(config, ground_truth_mode)

    rag_enabled = config['rag']['enabled']
    rag_path = config['rag']['rag_path']

    # Set the RAG log directory
    if rag_log_dir is None:
        rag_log_dir = os.path.join(write_result_file_path, "rag_result")

    experiment_config = ExperimentConfig(
        mode=observation_mode,
        global_reward_mode=global_reward_mode,
        planning_text_model=planning_text_model,
        global_reward_text_model=global_reward_text_model,
        ground_truth_mode=ground_truth_mode,
        single_task_name=single_task_name,
        config=config,
        ground_truth_data=ground_truth_data,
        write_result_file_path=write_result_file_path,
        record_time=record_time,
        file=file,
        rag_enabled=rag_enabled,
        rag_path=rag_path,
        screenshot_base_dir=screenshot_base_dir,
        rag_logging_enabled=rag_logging_enabled,
        rag_log_dir=rag_log_dir,
        rag_mode=rag_mode
    )

    await run_experiment(task_range, experiment_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the web agent with OpenAI Operator support.")
    parser.add_argument("--global_reward_mode",
                        choices=["dom_vision_reward", "dom_reward",
                                 "vision_reward", "no_global_reward"],
                        default="dom_reward", 
                        help="Choose the mode of global reward.")
    parser.add_argument("--index", type=str, default=-1)
    parser.add_argument("--single_task_name", type=str,
                        default="Find Dota 2 game and add all DLC to cart in steam.")
    parser.add_argument("--single_task_website", type=str,
                        default="about:blank", help="Website URL for single task mode")
    parser.add_argument("--snapshot", type=str, default="test/exp")
    parser.add_argument("--planning_text_model", type=str, default="computer-use-preview-2025-03-11")
    parser.add_argument("--global_reward_text_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--observation_mode", type=str, default="operator",
                        choices=["dom", "operator", "vision", "dom_v_desc", "vision_to_dom", "d_v"],
                        help="Choose the observation mode")
    parser.add_argument("--ground_truth_mode", action="store_true", 
                        help="Enable ground truth mode")
    parser.add_argument("--toml_path", type=str, default=None,
                        help="Path to TOML configuration file")
    parser.add_argument("--screenshot_base_dir", type=str, default=None,
                        help="Base directory for screenshots")
    parser.add_argument("--rag_logging_enabled", action="store_true",
                        help="Enable RAG logging for operator mode")
    parser.add_argument("--rag_log_dir", type=str, default=None,
                        help="Directory to store RAG logs (default: results_dir/rag_result)")
    parser.add_argument("--rag_mode", type=str, default="description",
                        choices=["description", "vision"],
                        help="RAG mode: description (use text descriptions) or vision (use visual examples)")

    args = parser.parse_args()

    asyncio.run(main(global_reward_mode=args.global_reward_mode,
                     planning_text_model=args.planning_text_model,
                     global_reward_text_model=args.global_reward_text_model,
                     single_task_name=args.single_task_name,
                     single_task_website=args.single_task_website,
                     raw_data_index=args.index,
                     observation_mode=args.observation_mode,
                     ground_truth_mode=args.ground_truth_mode,
                     toml_path=args.toml_path,
                     screenshot_base_dir=args.screenshot_base_dir,
                     rag_logging_enabled=args.rag_logging_enabled,
                     rag_log_dir=args.rag_log_dir,
                     rag_mode=args.rag_mode
                     )
                ) 
