#import numpy as np

#from contextlib import redirect_stdout
#import io
import sys
#import threading


##### Exlore call profiling

'''
research python profilers
make context manager

'''

#https://stackoverflow.com/questions/8315389/how-do-i-print-functions-as-they-are-called
#https://docs.python.org/2/library/inspect.html # for details
# https://docs.python.org/3/library/sys.html#sys.setprofile # for details

# for the timing part https://stackoverflow.com/a/28218696



#keep track by level, and associate each call with a level

#### The class / context
class profile:
    def __init__(self, file=sys.stdout):
        '''
        file: (str or file-like)


        dev note: only .seekable() file-likes are closed at teh end. Possibly
         future explicity close all but stdout/in/err
        '''
        self.level = 0

        if isinstance(file, str):
            open(file, 'w').close() # truncate file
            file = open(file, 'at')
        self.file = file

    def tracefunc(self,frame, event, arg):
        if (event == "call"):
            self.level += 1
            print("-" * self.level + "> call function", file=self.file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
        #if (event == "c_call"):
        #    self.level += 1
        #    print("-" * self.level + "> c_call function", file=self.file)
        #    print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
        elif event == "return":
            print("<" + "-" * self.level, "exit function",file=self.file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
            self.level -= 1
            #print(<time spent in func since call exclued>)
        #return self.tracefunc # why???


    def __enter__(self):
        sys.setprofile(self.tracefunc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setprofile(None)
        if self.file.seekable(): # excludes sys.stdout
            self.file.close()



##### The base fuc

def tracefunc(frame, event, arg, level=[0],file=sys.stdout):
    '''

    Questions:
    - why is indent not a scalar? A: VERY CLEVER because it saves the value in memory 
    that can be retireved each time the func is called even tho its not a 
    class

    Issue: add filter to only print calls to certain modules

    Issue: add a timing functionality
    '''
    
    with open('temp.txt', 'at') as file:
        print('tracefunc activated', file=file)
        if (event == "call"):
            level[0] += 1
            print("-" * level[0] + "> call function", file=file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=file)
        #elif (event == "c_call"):
        #    level[0] += 1
        #    print("-" * level[0] + "> c_call function", file=file)
        #    print(frame.f_code.co_filename, frame.f_code.co_name,file=file)
        elif event == "return":
            print("<" + "-" * level[0], "exit function",file=file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=file)
            level[0] -= 1
        #print('#\n'*20, file=file)
            #print(<time spent in func since call exclued>)
        #return tracefunc # why???





##########
#if __name__ == '__main__':






if False:

    ######################
    import datetime as dt
    import timeit

    class TimeManager:
        """Context Manager used with the statement 'with' to time some execution.

        Example:

        with TimingManager() as t:
        # Code to time
        """

        self.clock = timeit.default_timer

        def __enter__(self):
            """
            """
            self.start = self.clock()
            self.log('\n=> Start Timing: {}')

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            """
            self.endlog()

            return False


        def log(self, s, elapsed=None):
            """Log current time and elapsed time if present.
            :param s: Text to display, use '{}' to format the text with
                the current time.
            :param elapsed: Elapsed time to display. Dafault: None, no display.
            """
            print(s.format(self._secondsToStr(self.clock())))

            if(elapsed is not None):
                print('Elapsed time: {}\n'.format(elapsed))

        def endlog(self):
            """Log time for the end of execution with elapsed time.
            """
            self.log('=> End Timing: {}', self.now())

        def now(self):
            """Return current elapsed time as hh:mm:ss string.
            :return: String.
            """
            return str(dt.timedelta(seconds = self.clock() - self.start))

        def _secondsToStr(self, sec):
            """Convert timestamp to h:mm:ss string.
            :param sec: Timestamp.
            """
            return str(dt.datetime.fromtimestamp(sec))


