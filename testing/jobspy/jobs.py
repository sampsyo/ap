#!/usr/bin/env python
from __future__ import with_statement
import threading, sys
from subprocess import Popen, PIPE
from optparse import OptionParser

def dbgprint(txt):
    if debug:
        print >>sys.stderr, 'DEBUG: ' + str(txt)

class Executor(threading.Thread):
    """A consumer responsible for executing commands on one remote host."""
    
    def __init__(self, host, user=None):
        super(Executor, self).__init__()
        self.host = host
        self.user = user
    
    def log(self):
        """Print the results of a command execution when it finishes."""
        print >>sys.stderr, '[[ ' + ((self.user + '@') if self.user else '') + \
                self.host + ': ' + self.cmd + ' ]]'
        print self.stdout
        dbgprint('executed; ' + str(len(remaining_cmds)) + ' commands remain')
    
    def run(self):
        while len(remaining_cmds) > 0:
            # consume a command
            self.cmd = remaining_cmds.pop(0)
        
            # make the SSH command & arguments
            if self.user:
                hostarg = self.user + '@' + self.host
            else:
                hostarg = self.host
            args = [ssh_path,
                        '-n', # no input
                        '-o', 'StrictHostKeyChecking=no', # security issue
                        hostarg, self.cmd]
    
            # run the process
            self.process = Popen(args, stdout=PIPE)
            (self.stdout, self.stderr) = self.process.communicate()
        
            # report our output
            self.log()
            
            with finishcond:
                finishcond.notify()

def run(hosts, cmds):
    """Run the commands on the hosts. Main driver function."""
    global remaining_cmds, active_executors, finishcond
    remaining_cmds = cmds
    executors = []
    finishcond = threading.Condition()
    
    # schedule initial set of tasks
    while hosts:
        (host, user) = hosts.pop()
        ex = Executor(host, user)
        ex.start()
        executors.append(ex)
    
    # wait for consumers to finish
    while len(remaining_cmds) > 0:
        try:
            with finishcond:
                finishcond.wait()
        except KeyboardInterrupt:
            dbgprint('got keyboard interrupt')
            sys.exit()
        dbgprint('notified; ' + str(len(remaining_cmds)) + ' commands remain')
    
    dbgprint('all commands finished')

if __name__ == '__main__':
    # read options
    usage = """usage: %prog [options] hosts commands"""
    op = OptionParser(usage=usage)
    op.add_option('-v', dest='debug', action='store_true', default=False,
                  help='output debug information to stderr')
    op.add_option('--ssh_path', dest='ssh_path', default='/usr/bin/ssh',
                  help='use SSH executable at PATH (default %default)',
                  metavar='PATH', type='string')
    (options, args) = op.parse_args()
    global ssh_path, debug
    ssh_path = options.ssh_path
    debug = options.debug
    
    # read file arguments
    if len(args) < 2:
        op.error('hosts and commands files are required')
    elif len(args) > 2:
        op.error('too many arguments')
    hostsfile = args[0]
    cmdsfile = args[1]
    
    # read list of (host,user) pairs
    hosts = []
    hostsfh = open(hostsfile)
    for line in hostsfh:
        chunks = line.split()
        if len(chunks) >= 1:
            host = chunks[0]
            if len(chunks) >= 2:
                user = chunks[1]
            else:
                user = None
            hosts.append((host, user))
    hostsfh.close()
    
    # read list of commands to run
    cmds = []
    cmdsfh = open(cmdsfile)
    for line in cmdsfh:
        line = line.strip()
        if line:
            cmds.append(line)
    cmdsfh.close()
    
    # schedule 'em!
    run(hosts, cmds)