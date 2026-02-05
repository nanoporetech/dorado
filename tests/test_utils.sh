# Shared bash functions used in testing

# Print a title to separate tests in logs.
title() {

    # Record and disable xtrace to reduce noise
    local had_xtrace=0
    case $- in
        *x*) had_xtrace=1;
        set +x ;;
    esac

    # Create a line of `=` 80 characters wide
    local n_cols=80
    local line=$(printf '%*s' "$n_cols" '' | tr ' ' '=')

    echo ""
    echo ""
    echo "$line"
    echo "$*"
    echo "$line"

    # Reset xtrace to initial state
    if [ "$had_xtrace" -eq 1 ]; then
        set -x
    fi
}
