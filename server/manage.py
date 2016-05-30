#!/usr/bin/env python
import os
import sys

sys.path.append('../')
sys.path.append('../classifier')

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "blog_telegram.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
