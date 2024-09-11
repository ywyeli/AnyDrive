import threading
from time import sleep
from threading import Condition
class AutoPlay(threading.Thread):
    def __init__(self, func, sleep_time=2):
        super().__init__()
        self.func = func
        self.sleep_time = sleep_time
        self.cond = Condition()
        self.__should_run = False
        self.__should_exit = False
    
    def run(self):
        while not self.__should_exit:
            self.cond.acquire()
            while not self.__should_run and not self.__should_exit:
                # print('waiting', self.__should_exit,self.__should_run)
                self.cond.wait()
            # print("running in thread", self.__should_exit,self.__should_run)
            if self.__should_run:
                self.func()
                sleep(self.sleep_time)
            self.cond.release()
        # print('exiting')
    
    def kill(self):
        self.__should_exit = True
        self.__should_run = False
        self.cond.notify_all()

    def change_run_state(self):
        self.__should_run = not self.__should_run
        self.cond.acquire()
        self.cond.notify()
        self.cond.release()

    def set_sleep_time(self, sleep_time):
        self.sleep_time = sleep_time
