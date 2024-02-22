#import numpy as np

#from contextlib import redirect_stdout
#import io
import sys
import datetime as dt
import timeit
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

# https://docs.python.org/3/library/debug.html see for overview of profiling in base python

#keep track by level, and associate each call with a level

#### The class / context
class profile:
    '''

    timing note: some implrecision due to this being system time and 
    due to the profiling code (like print() etc) also taking up time 
    that is counted toward the original code.... maybe assign the 
    differences as a var and adjust in "call" not return?
    
    Issue: set pruning so that it only prints calls down to the nth level

    Issue: think of a more effcient way to record single level and all 
    contained levels eleapsed times. not two howl list with double pops and appends

    Issue: allow sum of time at each level i.e. time on level and time on 
    level and every level beneath

    Issue: save sys.stdout and replace with file, during call so that 
    I don't have to specify file= in every print call

    Issue: kill len(self.level_start) by adding some extra enter at 
    start before __enter__ exit
    '''
    
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

        self.clock = timeit.default_timer
        # level_start use the 'stack' concept
        self.level_start = []
        self.level_start_adj = [] # adjusted start times

    def tracefunc(self,frame, event, arg):
        if (event == "call"):
            self.level += 1
            print("-" * self.level + "> call function", file=self.file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
            start = self.clock()
            self.level_start.append(start)
            self.level_start_adj.append(start)
        #if (event == "c_call"):
        #    self.level += 1
        #    print("-" * self.level + "> c_call function", file=self.file)
        #    print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
        elif event == "return":
            print("<" + "-" * self.level, "exit function",file=self.file)
            print(frame.f_code.co_filename, frame.f_code.co_name,file=self.file)
            # for profile start on exit - is try catch faster?
            if len(self.level_start) > 0:
                #elapsed = self.clock() - (self.level_start.pop() + self.total_elapsed) # failed attempt to caputre both
                end = self.clock()
                elapsed_net = end - self.level_start.pop() # old but beautiful
                elapsed = end - self.level_start_adj.pop()
                # remove time spent on child level from all parent levels
                self.level_start_adj = [ts + elapsed for ts in self.level_start_adj] # old but beautiful
                t_net = dt.timedelta(seconds=elapsed_net)
                t = dt.timedelta(seconds=elapsed)
                print(' -- took %s gross; %s exclusive (hh:mm:ss)' 
                %(t_net, t), file=self.file)
                self.total_elapsed += elapsed                    
            self.level -= 1
            #print(<time spent in func since call exclued>)
        #return self.tracefunc # why???

    def _secs_to_str(self, sec):
        """Convert timestamp to h:mm:ss string.
        :param sec: Timestamp.
        """
        return str(dt.datetime.fromtimestamp(sec))



    def __enter__(self):
        sys.setprofile(self.tracefunc)
        self.total_elapsed = 0
        self.total_net = self.clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setprofile(None)
        self.total_net = self.clock() - self.total_net 
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


