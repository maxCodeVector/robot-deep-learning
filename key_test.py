import tty
import sys
import termios
import threading


class Job(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(Job, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()     # 用于暂停线程的标识
        self.__flag.set()       # 设置为True
        self.__running = threading.Event()      # 用于停止线程的标识
        self.__running.set()      # 将running设置为True

        self.ins = 'w'

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()      # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
            print(self.ins, end='')
            # time.sleep(1)

    def pause(self):
        self.__flag.clear()     # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()    # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()       # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()        # 设置为False


def join_control(ins):
    for i in range(5):
        print("You pressed", ins)


if __name__ == '__main__':

    t = Job()
    t.setDaemon(True)
    t.start()

    orig_settings = termios.tcgetattr(sys.stdin)

    tty.setcbreak(sys.stdin)
    x = 0
    while x != chr(27):  # ESC
        x = sys.stdin.read(1)[0]
        t.pause()
        join_control(x)
        t.resume()

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
    print("now exit with coed 0")
    exit(0)
