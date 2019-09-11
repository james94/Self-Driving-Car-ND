from subprocess import Popen, PIPE
import os

# FileSystemCli contains Linux file system commands
class FileSystemCli:
    def __init__(self, os_flavor = None):
        # process
        self.os_flavor_m = os_flavor
        
    def listdir_shell(self, path, *lsargs):
        """
            List the directory contents of files and directories
        """
        if self.os_flavor_m == "Linux":
            p = Popen(("ls", path) + lsargs, shell=False, stdout=PIPE, close_fds = True, encoding = "utf8")
            # loops through dir path, reads each line from stdout and removes
            # each trailing character "\n" from each string in the list, 
            # then returns a list of elements of type string
            return [path.rstrip("\n") for path in p.stdout.readlines()]
        else:
            print("OS Flavor not supported")