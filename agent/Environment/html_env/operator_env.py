"""
OpenAI Operator Environment
Specifically dealing with environment interactions for OpenAI operators
"""

import os
import base64
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext, Error as PWError
from sanic.log import logger
from beartype import beartype

from .operator_actions import (
    OperatorAction, OperatorActionExecutor, OperatorResponseParser,
    OperatorActionFactory, OperatorActionType
)


class OperatorEnvironment:
    """OpenAI Operator Environment"""
    
    def __init__(self, 
                 headless: bool = True,
                 slow_mo: int = 50,
                 viewport_width: int = 1280,
                 viewport_height: int = 720,
                 save_trace_enabled: bool = True,
                 screenshot_dir: str = "screenshots"):
        """
        初始化Operator环境
        
        Args:
            headless: 是否无头模式
            slow_mo: 操作延迟（毫秒）
            viewport_width: 视口宽度
            viewport_height: 视口高度
            save_trace_enabled: 是否保存追踪信息
            screenshot_dir: 截图保存目录
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.save_trace_enabled = save_trace_enabled
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Playwright
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Action executor
        self.action_executor = None
        
        # 追踪信息
        self.trace_data = []
        self.step_count = 0
        
    async def start(self):
        """Start environment"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
            self.context = await self.browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )
            self.page = await self.context.new_page()
            
            # action executor
            self.action_executor = OperatorActionExecutor(self.page)
            
            logger.info("OperatorEnvironment started successfully")
            
        except Exception as e:
            logger.error(f"Error starting OperatorEnvironment: {e}")
            raise
    
    async def close(self):
        """关闭环境"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("OperatorEnvironment closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing OperatorEnvironment: {e}")
    
    async def navigate_to(self, url: str, max_retries: int = 3):
        """
        导航到指定URL，使用多重策略确保稳定性
        
        Args:
            url: 目标URL
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🌐 Navigating to {url} (attempt {attempt + 1}/{max_retries})")
                
                # 1：导航到页面，使用较长的超时时间
                await self.page.goto(url, timeout=45000, wait_until="domcontentloaded")
                logger.info("✅ Page loaded (DOM content ready)")
                
                # 2：尝试等待网络空闲，但使用较短的超时
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=8000)
                    logger.info("✅ Network idle achieved")
                    break
                    
                except Exception as e:
                    if "Timeout" in str(e):
                        logger.warning(f"⚠️  Network idle timeout, trying alternative strategy...")
                        
                        # 3：如果网络空闲失败，尝试等待加载完成
                        try:
                            await self.page.wait_for_load_state("load", timeout=5000)
                            logger.info("✅ Page load state achieved")
                            break
                            
                        except Exception as e2:
                            if "Timeout" in str(e2):
                                logger.warning(f"⚠️  Load state timeout, using minimal wait...")
                                
                                # 4：简单等待
                                await asyncio.sleep(3)
                                
                                # 检查页面是否至少部分可用
                                try:
                                    title = await self.page.title()
                                    if title and title.strip():
                                        logger.info(f"✅ Page accessible with title: '{title}'")
                                        break
                                    else:
                                        raise Exception("Page title is empty")
                                except:
                                    if attempt < max_retries - 1:
                                        logger.warning(f"❌ Page not properly loaded, retrying...")
                                        continue
                                    else:
                                        raise Exception("Page failed to load after all strategies")
                            else:
                                raise e2
                    else:
                        raise e
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    logger.warning(f"❌ Navigation attempt {attempt + 1} failed: {e}")
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ All navigation attempts failed for {url}: {e}")
                    raise Exception(f"Failed to navigate to {url} after {max_retries} attempts: {e}")
        
        # 导航成功后，等待页面完全准备就绪
        logger.info("🔄 Ensuring page is fully ready...")
        page_ready = await self.wait_for_page_ready()
        if not page_ready:
            logger.warning("⚠️  Page readiness check failed, but continuing...")
        
        logger.info(f"🎯 Successfully navigated to {url}")
    
    async def check_network_health(self, url: str) -> bool:
        """
        检查网络健康状态和目标URL的可达性
        
        Args:
            url: 要检查的URL
            
        Returns:
            bool: 网络是否健康
        """
        try:
            # 尝试快速访问目标域名
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            # 使用简单的HTTP请求测试连接
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.head(f"https://{domain}", allow_redirects=True) as response:
                    if response.status < 500:  # 允许4xx错误，只要服务器响应
                        logger.info(f"✅ Network health check passed for {domain}")
                        return True
                    else:
                        logger.warning(f"⚠️  Server error {response.status} for {domain}")
                        return False
                        
        except Exception as e:
            logger.warning(f"⚠️  Network health check failed for {url}: {e}")
            return False
    
    async def wait_for_page_ready(self, timeout: int = 15000) -> bool:
        """
        等待页面真正准备就绪，包括动态内容
        
        Args:
            timeout: 超时时间（毫秒）
            
        Returns:
            bool: 页面是否准备就绪
        """
        try:
            logger.info("⏳ Waiting for page to be fully ready...")
            
            # 策略1：等待DOM和初始资源加载
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout // 3)
            
            # 策略2：等待基本元素出现
            try:
                # 等待页面body元素完全加载
                await self.page.wait_for_selector("body", timeout=5000)
                
                # 等待一小段时间让动态内容加载
                await asyncio.sleep(2)
                
                # 检查页面是否有基本内容
                content_length = await self.page.evaluate("document.body.innerText.length")
                if content_length > 100:  # 页面有足够内容
                    logger.info(f"✅ Page ready with {content_length} characters of content")
                    return True
                    
            except Exception:
                pass
            
            # 策略3：如果上述失败，尝试更宽松的条件
            try:
                # 检查页面标题是否存在
                title = await self.page.title()
                if title and len(title.strip()) > 0:
                    logger.info(f"✅ Page ready with title: '{title}'")
                    return True
            except Exception:
                pass
            
            # 策略4：最后的备选方案
            logger.warning("⚠️  Using fallback readiness check...")
            await asyncio.sleep(3)
            return True  # 假设页面已准备就绪
            
        except Exception as e:
            logger.warning(f"⚠️  Page readiness check failed: {e}")
            return False
    
    async def take_screenshot(self, filename: Optional[str] = None) -> str:
        """
        截图并返回base64编码
        
        Args:
            filename: 可选的文件名
            
        Returns:
            base64编码的截图
        """
        try:
            # 移除强制滚动到顶部的操作，让截图反映当前页面状态
            # await self._safe_scroll_top()
            
            # 生成文件名
            if filename is None:
                filename = f"step_{self.step_count:03d}.png"
            
            screenshot_path = self.screenshot_dir / filename
            
            # 截图
            screenshot_data = await self.page.screenshot(
                path=str(screenshot_path),
                full_page=False,
                type="png"
            )
            
            # 转换为base64
            screenshot_b64 = base64.b64encode(screenshot_data).decode()
            
            logger.info(f"Screenshot saved: {screenshot_path}")
            
            return screenshot_b64
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            raise
    
    async def _safe_scroll_top(self):
        """安全滚动到顶部"""
        for _ in range(2):  # 最多重试一次
            try:
                await self.page.evaluate("() => window.scrollTo(0,0)")
                return
            except PWError as e:
                if "Execution context was destroyed" in str(e):
                    await self.page.wait_for_load_state("load")
                    continue
                raise
    
    async def execute_operator_actions(self, actions: List[OperatorAction]) -> bool:
        """
        执行一系列Operator actions
        
        Args:
            actions: 要执行的操作列表
            
        Returns:
            是否成功执行所有操作
        """
        if not actions:
            return True
            
        try:
            for action in actions:
                logger.info(f"Executing operator action: {action.type}")
                
                if action.type != OperatorActionType.SCREENSHOT:
                    await self.action_executor.execute_action(action)
                
                # 记录追踪信息
                if self.save_trace_enabled:
                    self.trace_data.append({
                        "step": self.step_count,
                        "action": action.to_dict(),
                        "timestamp": time.time(),
                        "url": self.page.url
                    })
                
                self.step_count += 1
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing operator actions: {e}")
            return False
    
    async def get_current_state(self) -> Dict[str, Any]:
        """获取当前页面状态"""
        try:
            return {
                "url": self.page.url,
                "title": await self.page.title(),
                "viewport": {
                    "width": self.viewport_width,
                    "height": self.viewport_height
                },
                "step_count": self.step_count
            }
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {}
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """获取OpenAI Operator工具规范"""
        return {
            "type": "computer_use_preview",
            "display_width": self.viewport_width,
            "display_height": self.viewport_height,
            "environment": "browser"
        }
    
    def get_trace_data(self) -> List[Dict[str, Any]]:
        """获取追踪数据"""
        return self.trace_data.copy()
    
    def clear_trace_data(self):
        """清除追踪数据"""
        self.trace_data.clear()
        self.step_count = 0
    
    async def wait_for_load(self, timeout: int = 10000):
        """等待页面加载完成"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception as e:
            logger.warning(f"Wait for load timeout: {e}")


class OperatorEnvironmentManager:
    """Operator环境管理器"""
    
    def __init__(self):
        self.environments = {}
    
    async def create_environment(self, 
                               env_id: str,
                               **kwargs) -> OperatorEnvironment:
        """创建新的Operator环境"""
        if env_id in self.environments:
            await self.close_environment(env_id)
        
        env = OperatorEnvironment(**kwargs)
        await env.start()
        self.environments[env_id] = env
        
        return env
    
    async def get_environment(self, env_id: str) -> Optional[OperatorEnvironment]:
        """获取指定的环境"""
        return self.environments.get(env_id)
    
    async def close_environment(self, env_id: str):
        """关闭指定的环境"""
        if env_id in self.environments:
            await self.environments[env_id].close()
            del self.environments[env_id]
    
    async def close_all_environments(self):
        """关闭所有环境"""
        for env_id in list(self.environments.keys()):
            await self.close_environment(env_id)


operator_env_manager = OperatorEnvironmentManager() 