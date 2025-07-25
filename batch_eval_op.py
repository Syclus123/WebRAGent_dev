"""
This is a batch test script specifically for OpenAI Operator mode.
This release adds the following features:
1. Support batch evaluation of OpenAI Operator mode (model :computer-use-preview-2025-03-11)
2. Support pure visual observation mode
3. Support RAG logging and custom storage paths

Usage:
    sudo yum install -y xorg-x11-server-Xvfb
    xvfb-run -a python batch_eval_op.py
    
    Ubuntu/Debian User:
    sudo apt-get update
    sudo apt-get install -y xvfb

    xvfb-run -a python batch_eval_op.py --global_reward_mode dom_reward --global_reward_text_model gpt-4.1 --snapshot test/exp --output_log test/exp/batch_operator_log.txt --rag_logging_enabled --rag_log_dir test/exp/rag_logs
"""

#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
import time
from pathlib import Path

def load_tasks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def run_single_operator_task(task, current_idx, args):
    task_name = task["confirmed_task"]
    website = task.get("website", "about:blank")
    task_id = task.get("task_id", f"task_{current_idx}")
    
    # eval_op.py params
    command = [
        "python", "eval_op.py",
        "--observation_mode", "operator",
        "--global_reward_mode", args.global_reward_mode,
        "--index", str(current_idx),
        "--single_task_name", task_name,
        "--single_task_website", website,
        "--snapshot", args.snapshot,
        "--planning_text_model", args.planning_text_model,
        "--global_reward_text_model", args.global_reward_text_model,
        "--screenshot_base_dir", args.snapshot
    ]
    
    if args.ground_truth_mode:
        command.append("--ground_truth_mode")
    
    if args.toml_path:
        command.extend(["--toml_path", args.toml_path])
    
    # RAG logging
    if args.rag_logging_enabled:
        command.append("--rag_logging_enabled")
    
    # RAG dir
    if args.rag_log_dir:
        command.extend(["--rag_log_dir", args.rag_log_dir])
    
    print(f"\n{'='*80}")
    print(f"🤖 Operator任务 [{current_idx}]: {task_name}")
    print(f"🌐 网站: {website}")
    print(f"🔧 任务ID: {task_id}")
    print(f"📱 模型: {args.planning_text_model}")
    print(f"📝 RAG日志: {'启用' if args.rag_logging_enabled else '禁用'}")
    if args.rag_logging_enabled and args.rag_log_dir:
        print(f"📂 RAG日志目录: {args.rag_log_dir}")
    print(f"{'='*80}")
    
    try:
        # use headless mode
        if args.use_xvfb:
            full_command = ["xvfb-run", "-a"] + command
        else:
            full_command = command
            
        subprocess.run(full_command, check=True)
        print(f"✅ 任务完成: {task_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 任务失败: {task_name}")
        print(f"🔥 错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='OpenAI Operator Mode Batch Evaluation')
    parser.add_argument('--json_path', type=str, default='data/Online-Mind2Web/72exp30.json',
                        help='JSON任务文件路径')
    parser.add_argument('--global_reward_mode', type=str, default='dom_reward',
                        help='全局奖励模式: dom_reward/no_global_reward/dom_vision_reward')
    parser.add_argument('--index', type=int, default=-1,
                        help='任务索引')
    parser.add_argument('--snapshot', type=str, default='test/exp',
                        help='截图目录')
    parser.add_argument('--planning_text_model', type=str, default='computer-use-preview-2025-03-11',
                        help='规划文本模型: computer-use-preview-2025-03-11/gpt-4.1')
    parser.add_argument('--global_reward_text_model', type=str, default='gpt-4.1',
                        help='全局奖励文本模型: gpt-4.1/gpt-4o-mini')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='开始任务的索引')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束任务的索引（不包含）')
    parser.add_argument('--delay', type=int, default=10,
                        help='任务间延迟时间（秒）')
    parser.add_argument('--output_log', type=str, default='test/exp/batch_operator_log.txt',
                        help='输出日志文件')
    parser.add_argument('--ground_truth_mode', action="store_true",
                        help='启用真实标准模式')
    parser.add_argument('--toml_path', type=str, default=None,
                        help='TOML配置文件路径')
    parser.add_argument('--use_xvfb', action="store_true", default=True,
                        help='使用xvfb运行（默认启用）')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='每个任务的最大重试次数')
    parser.add_argument('--rag_logging_enabled', action="store_true", default=False,
                        help='启用RAG日志记录')
    parser.add_argument('--rag_log_dir', type=str, default=None,
                        help='RAG日志文件的输出目录')
    
    args = parser.parse_args()
    
    # load tasks
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"❌ 错误: 文件不存在 - {json_path}")
        return
    
    tasks = load_tasks(json_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    
    total_tasks = end_idx - start_idx
    successful_tasks = 0
    failed_tasks = []
    
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    os.makedirs(args.snapshot, exist_ok=True)
    
    img_screenshots_dir = os.path.join(args.snapshot, "img_screenshots")
    logs_dir = os.path.join(args.snapshot, "logs")
    os.makedirs(img_screenshots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    if args.output_log == 'results_operator/batch_exp1/logs/batch_operator_log.txt':
        args.output_log = os.path.join(logs_dir, 'batch_operator_log.txt')
    
    # init log file
    with open(args.output_log, 'w', encoding='utf-8') as log_file:
        log_file.write(f"🚀 OpenAI Operator批量任务开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"📊 总任务数: {total_tasks}\n")
        log_file.write(f"🤖 使用模型: {args.planning_text_model}\n")
        log_file.write(f"📁 结果目录: {args.snapshot}\n")
        log_file.write(f"📸 截图目录: {img_screenshots_dir}\n")
        log_file.write(f"⏱️  任务间延迟: {args.delay}秒\n")
        log_file.write(f"🔄 最大重试次数: {args.max_retries}\n")
        log_file.write(f"📝 RAG日志: {'启用' if args.rag_logging_enabled else '禁用'}\n")
        if args.rag_logging_enabled and args.rag_log_dir:
            log_file.write(f"📂 RAG日志目录: {args.rag_log_dir}\n")
        log_file.write("\n")
    
    print(f"🚀 开始OpenAI Operator批量任务评估")
    print(f"📊 任务范围: {start_idx} - {end_idx-1} (共{total_tasks}个任务)")
    print(f"🤖 使用模型: {args.planning_text_model}")
    print(f"📁 结果目录: {args.snapshot}")
    print(f"📸 截图目录: {img_screenshots_dir}")
    
    for i, task_data in enumerate(tasks[start_idx:end_idx]):
        current_idx = start_idx + i
        task_name = task_data["confirmed_task"]
        website = task_data.get("website", "about:blank")
        task_id = task_data.get("task_id", f"task_{current_idx}")

        with open(args.output_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f"📋 [{current_idx}/{len(tasks)}] 运行任务: {task_name}\n")
            log_file.write(f"🌐 网站: {website}\n")
            log_file.write(f"🔧 任务ID: {task_id}\n")
        
        # run task with retry
        success = False
        for attempt in range(args.max_retries + 1):
            if attempt > 0:
                print(f"🔄 重试第{attempt}次: {task_name}")
                with open(args.output_log, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"🔄 重试第{attempt}次\n")
            
            success = run_single_operator_task(task_data, current_idx, args)
            if success:
                break
            elif attempt < args.max_retries:
                retry_delay = min(30, args.delay * 2)  # 重试时增加延迟
                print(f"⏳ 等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
        
        if success:
            successful_tasks += 1
            print(f"✅ 任务成功: {task_name}")
        else:
            failed_tasks.append({
                "index": current_idx,
                "task_name": task_name,
                "task_id": task_id,
                "website": website
            })
            print(f"❌ 任务最终失败: {task_name}")
        
        with open(args.output_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f"📊 结果: {'成功' if success else '失败'}\n")
            if not success:
                log_file.write(f"❌ 经过{args.max_retries}次重试后仍然失败\n")
            log_file.write("\n")
        
        # wait for next task
        if i < total_tasks - 1:
            print(f"⏳ 等待{args.delay}秒后继续下一个任务...")
            time.sleep(args.delay)
    
    # final result
    success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    with open(args.output_log, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n🏁 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"📊 总任务数: {total_tasks}\n")
        log_file.write(f"✅ 成功任务数: {successful_tasks}\n")
        log_file.write(f"❌ 失败任务数: {len(failed_tasks)}\n")
        log_file.write(f"📈 成功率: {success_rate:.2f}%\n")
        
        if failed_tasks:
            log_file.write(f"\n❌ 失败任务详情:\n")
            for failed_task in failed_tasks:
                log_file.write(f"  - [{failed_task['index']}] {failed_task['task_name']} ({failed_task['task_id']})\n")
                log_file.write(f"    网站: {failed_task['website']}\n")
    
    print(f"\n{'='*80}")
    print(f"🏁 OpenAI Operator批量任务评估完成!")
    print(f"📊 总任务数: {total_tasks}")
    print(f"✅ 成功任务数: {successful_tasks}")
    print(f"❌ 失败任务数: {len(failed_tasks)}")
    print(f"📈 成功率: {success_rate:.2f}%")
    print(f"📝 详细日志: {args.output_log}")
    
    if failed_tasks:
        print(f"\n❌ 失败的任务:")
        for failed_task in failed_tasks:
            print(f"  - [{failed_task['index']}] {failed_task['task_name']}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()