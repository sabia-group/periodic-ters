#!/usr/bin/env bash

# Determine directory of this file (works for both bash and zsh when sourced)
_get_script_dir() {
    if [ -n "$BASH_VERSION" ]; then
        # bash: use BASH_SOURCE
        script="${BASH_SOURCE[0]}"
    elif [ -n "$ZSH_VERSION" ]; then
        # zsh: use ${(%):-%N} via eval so bash can still parse this file
        eval 'script="${(%):-%N}"'
    else
        # fallback: may not be correct when sourced, but better than nothing
        script="$0"
    fi

    # Resolve to absolute directory without relying on readlink -f (BSD/mac safe)
    cd "$(dirname "$script")" >/dev/null 2>&1 && pwd
}

ENV_BASE_DIR="$(_get_script_dir)"

prepend_path() {
    dir=$1
    var=$2

    # current value of the variable whose name is in $var
    eval "cur=\${$var}"

    # the path is a directory and is not yet included
    if [ -d "$dir" ] && [[ ":$cur:" != *":$dir:"* ]]; then
        eval "export $var=\"$dir\${cur:+\":\$cur\"}\""
    fi
}

prepend_path "$ENV_BASE_DIR" PYTHONPATH

unset ENV_BASE_DIR
unset -f _get_script_dir
unset -f prepend_path

