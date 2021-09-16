#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export prometheus_multiproc_dir=data
uwsgi run.ini