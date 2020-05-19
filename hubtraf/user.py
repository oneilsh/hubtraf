from enum import Enum, auto
import aiohttp
import socket
import uuid
import random
from yarl import URL
import asyncio
import async_timeout
import structlog
import time
import numpy as np

logger = structlog.get_logger()

class OperationError(Exception):
    pass


class User:
    class States(Enum):
        CLEAR = 1
        LOGGED_IN = 2
        SERVER_STARTED = 3
        KERNEL_STARTED = 4

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close() 

    def __init__(self, username, hub_url, login_handler, port=None, kernel=None):
        """
        A simulated JupyterHub user.

        username - name of the user.
        hub_url - base url of the hub.
        login_handler - a awaitable callable that will be passed the following parameters:
                            username
                            session (aiohttp session object)
                            log (structlog log object)
                            hub_url (yarl URL object)

                        It should 'log in' the user with whatever requests it needs to
                        perform. If no uncaught exception is thrown, login is considered
                        a success. 

                        Usually a partial of a generic function is passed in here.
        """
        self.username = username
        self.hub_url = URL(hub_url).with_port(port)

        self.state = User.States.CLEAR
        self.notebook_url = self.hub_url / 'user' / self.username

        self.log = logger.bind(
            username=username
        )
        self.login_handler = login_handler
        self.kernel = kernel

    async def login(self):
        """
        Log in to the JupyterHub.

        We only log in, and try to not start the server itself. This
        makes our testing code simpler, but we need to be aware of the fact this
        might cause differences vs how users normally use this.
        """
        # We only log in if we haven't done anything already!
        assert self.state == User.States.CLEAR

        start_time = time.monotonic()
        await self.login_handler(log=self.log, hub_url=self.hub_url, session=self.session, username=self.username)
        hub_cookie = self.session.cookie_jar.filter_cookies(self.hub_url).get('hub', None)
        if hub_cookie:
            self.log = self.log.bind(hub=hub_cookie.value)
        self.log.msg('Login: Complete', action='login', phase='complete', duration=time.monotonic() - start_time)
        self.state = User.States.LOGGED_IN

    # each user has a random timeout threshold, random refresh time interval, and a random amount of time they'll
    # wait before slowing down their refresh time by doubling it
    # refresh time is regenerated for each attempt based on the mean and sd values
    async def ensure_server(self, 
                            timeout_mean=1200,
                            timeout_sd=120,
                            spawn_refresh_time_mean=10, 
                            spawn_refresh_time_sd=2, 
                            doubling_time_mean=300,
                            doubling_time_sd=60):
        assert self.state == User.States.LOGGED_IN

        timeout = max(60, np.random.normal(timeout_mean, timeout_sd))
        doubling_time = max(120, np.random.normal(doubling_time_mean, doubling_time_sd))

        start_time = time.monotonic()
        self.log.msg(f'Server: Starting', action='server-start', phase='start')
        i = 0
        while True:
            i += 1
            self.log.msg(f'Server: Attempting to Starting', action='server-start', phase='attempt-start', attempt=i + 1)
            try:
                resp = await self.session.get(self.hub_url / 'hub/spawn')
            except Exception as e:
                self.log.msg('Server: Failed {}'.format(str(e)), action='server-start', attempt=i + 1, phase='attempt-failed', duration=time.monotonic() - start_time)
                continue
            # Check if paths match, ignoring query string (primarily, redirects=N), fragments
            target_url_tree = self.notebook_url / 'tree'
            # changed to .startswith(target_url_tree.path) in case default url is different (e.g. autonav to a user folder)
            if resp.url.scheme == target_url_tree.scheme and resp.url.host == target_url_tree.host and resp.url.path.startswith(target_url_tree.path):
                self.log.msg('Server: Started (Jupyter Notebook)', action='server-start', phase='complete', attempt=i + 1, duration=time.monotonic() - start_time)
                break
            target_url_lab = self.notebook_url / 'lab'
            if resp.url.scheme == target_url_lab.scheme and resp.url.host == target_url_lab.host and resp.url.path.startswith(target_url_lab.path):
                self.log.msg('Server: Started (JupyterLab)', action='server-start', phase='complete', attempt=i + 1, duration=time.monotonic() - start_time)
                break
            if time.monotonic() - start_time >= timeout:
                self.log.msg('Server: Timeout', action='server-start', phase='failed', duration=time.monotonic() - start_time)
                raise OperationError()
            if time.monotonic() - start_time >= doubling_time:
                self.log.msg('Server: Doubling Refresh Interval', action='server-start', phase='waiting', duration=time.monotonic() - start_time)
                spawn_refresh_time_mean = spawn_refresh_time_mean * 2
                spawn_refresh_time_sd = spawn_refresh_time_sd * 2
            # Always log retries, so we can count 'in-progress' actions
            self.log.msg('Server: Retrying', action='server-start', phase='attempt-complete', duration=time.monotonic() - start_time, attempt=i + 1)
            refresh_wait = max(5, np.random.normal(spawn_refresh_time_mean, spawn_refresh_time_sd))
            await asyncio.sleep(refresh_wait)
        
        self.state = User.States.SERVER_STARTED

    async def stop_server(self):
        assert self.state == User.States.SERVER_STARTED
        self.log.msg('Server: Stopping', action='server-stop', phase='start')
        start_time = time.monotonic()
        try:
            resp = await self.session.delete(
                self.hub_url / 'hub/api/users' / self.username / 'server',
                headers={'Referer': str(self.hub_url / 'hub/')}
            )
        except Exception as e:
            self.log.msg('Server: Failed {}'.format(str(e)), action='server-stop', phase='failed', duration=time.monotonic() - start_time)
            raise OperationError()
        if resp.status != 202 and resp.status != 204:
            self.log.msg('Server: Stop failed', action='server-stop', phase='failed', extra=str(resp), duration=time.monotonic() - start_time)
            raise OperationError()
        self.log.msg('Server: Stopped', action='server-stop', phase='complete', duration=time.monotonic() - start_time)
        self.state = User.States.LOGGED_IN

    async def start_kernel(self):
        assert self.state == User.States.SERVER_STARTED

        self.log.msg('Kernel: Starting', action='kernel-start', phase='start')
        start_time = time.monotonic()

        try:
            if self.kernel:
                resp = await self.session.post(self.notebook_url / 'api/kernels', headers={'X-XSRFToken': self.xsrf_token}, json = {'name': self.kernel})
            else:
                resp = await self.session.post(self.notebook_url / 'api/kernels', headers={'X-XSRFToken': self.xsrf_token})
        except Exception as e:
            self.log.msg('Kernel: Start failed {}'.format(str(e)), action='kernel-start', phase='failed', duration=time.monotonic() - start_time)
            raise OperationError()
        
        if resp.status != 201:
            self.log.msg('Kernel: Ststart failed', action='kernel-start', phase='failed', extra=str(resp), duration=time.monotonic() - start_time)
            raise OperationError()
        self.kernel_id = (await resp.json())['id']
        self.log.msg('Kernel: Started', action='kernel-start', phase='complete', duration=time.monotonic() - start_time)
        self.state = User.States.KERNEL_STARTED

    @property
    def xsrf_token(self):
        notebook_cookies = self.session.cookie_jar.filter_cookies(self.notebook_url)
        assert '_xsrf' in notebook_cookies
        xsrf_token = notebook_cookies['_xsrf'].value
        return xsrf_token

    async def stop_kernel(self):
        assert self.state == User.States.KERNEL_STARTED

        self.log.msg('Kernel: Stopping', action='kernel-stop', phase='start')
        start_time = time.monotonic()
        try:
            resp = await self.session.delete(self.notebook_url / 'api/kernels' / self.kernel_id, headers={'X-XSRFToken': self.xsrf_token})
        except Exception as e:
            self.log.msg('Kernel:Failed Stopped {}'.format(str(e)), action='kernel-stop', phase='failed', duration=time.monotonic() - start_time)
            raise OperationError()

        if resp.status != 204:
            self.log.msg('Kernel:Failed Stopped {}'.format(str(resp.url)), action='kernel-stop', phase='failed', duration=time.monotonic() - start_time)
            raise OperationError()

        self.log.msg('Kernel: Stopped', action='kernel-stop', phase='complete', duration=time.monotonic() - start_time)
        self.state = User.States.SERVER_STARTED

    def request_execute_code(self, msg_id, code):
        return {
            "header": {
                "msg_id": msg_id,
                "username": self.username,
                "msg_type": "execute_request",
                "version": "5.2"
            },
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": True,
                "stop_on_error": True
            },
            "buffers": [],
            "parent_header": {},
            "channel": "shell"
        }

    # execute_timeout is not used?
    async def assert_code_output(self, code, output, execute_timeout, repeat_time_seconds):
        channel_url = self.notebook_url / 'api/kernels' / self.kernel_id / 'channels'
        self.log.msg('WS: Connecting', action='kernel-connect', phase='start')
        is_connected = False
        success = False
        try:
            async with self.session.ws_connect(channel_url) as ws:
                is_connected = True
                self.log.msg('WS: Connected', action='kernel-connect', phase='complete')
                start_time = time.monotonic()
                iteration = 0
                self.log.msg('Code Execute: Started', action='code-execute', phase='start')
                while time.monotonic() - start_time < repeat_time_seconds:
                    exec_start_time = time.monotonic()
                    iteration += 1
                    msg_id = str(uuid.uuid4())
                    if not success:
                        await ws.send_json(self.request_execute_code(msg_id, code))
                        async for msg_text in ws:
                            if msg_text.type != aiohttp.WSMsgType.TEXT:
                                self.log.msg(
                                    'WS: Unexpected message type', 
                                    action='code-execute', phase='failure', 
                                    iteration=iteration,
                                    message_type=msg_text.type, message=str(msg_text), 
                                    duration=time.monotonic() - exec_start_time
                                )
                                raise OperationError()
    
                            msg = msg_text.json()
    
                            if 'parent_header' in msg and msg['parent_header'].get('msg_id') == msg_id:
                                # These are responses to our request
                                if msg['channel'] == 'iopub':
                                    response = None
                                    if msg['msg_type'] == 'execute_result':
                                        response = msg['content']['data']['text/plain']
                                    elif msg['msg_type'] == 'stream':
                                        response = msg['content']['text']
                                    if response:
                                        if response == output:
                                            self.log.msg('Execute Success', response=response, output=output)
                                            success = True
                                        else:
                                            self.log.msg('Response does NOT equal output, trying again', response=response, output=output)
                                        duration = time.monotonic() - exec_start_time
                                        break
                        # Sleep a random amount of time between 1 and 4s, so we aren't busylooping
                        await asyncio.sleep(random.uniform(1, 4))
    
                self.log.msg(
                    'Code Execute: complete', 
                    action='code-execute', phase='complete', 
                    duration=duration, iteration=iteration
                )
        except Exception as e:
            if type(e) is OperationError:
                raise
            if is_connected:
                self.log.msg('Code Execute: Failed {}'.format(str(e)), action='code-execute', phase='failure')
            else:
                self.log.msg('WS: Failed {}'.format(str(e)), action='kernel-connect', phase='failure')
            raise OperationError()
