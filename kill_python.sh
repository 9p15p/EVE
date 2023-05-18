#!/bin/bash
ps -ef | grep /xmem/bin/python | grep -v grep | awk '{print $2}' | xargs kill -9