#!/bin/bash
# ==============================================================================
# SSH Tunnel Proxy for Leonardo Compute Nodes
#
# Creates a SOCKS5 proxy via SSH dynamic port forwarding from compute → login.
# Based on Marianna's start_proxy_tunnel_lrdn.sh.
#
# Prerequisites:
#   1. Generate SSH key on login node:
#      ssh-keygen -t ed25519 -f ~/.ssh/leonardo_tunnel
#      (passphrase: 12345 — Leonardo enforces a passphrase)
#   2. Add pubkey to authorized_keys:
#      cat ~/.ssh/leonardo_tunnel.pub >> ~/.ssh/authorized_keys
#   3. Set SSH_KEY env var (or use default path)
#
# Usage (from sbatch script):
#   CMD_PREFIX=$(bash eval/leonardo/start_proxy_tunnel.sh)
#   $CMD_PREFIX python my_script.py   # runs through proxy
#
# The ONLY stdout output is the proxychains command prefix.
# All diagnostic messages go to stderr.
# ==============================================================================

set -e

NODE_HOST=$(hostname -s)

# Determine login node based on compute node hostname
if [[ $NODE_HOST == lrdn* ]]; then
    LOGIN_NODE="login05"
else
    echo "ERROR: Not a Leonardo compute node (hostname: $NODE_HOST)" >&2
    exit 1
fi

TUNNEL_PORT="${TUNNEL_PORT:-27003}"

# SSH key for intra-cluster tunneling
# NOTE: Leonardo FORCES a passphrase on SSH keys. Default: 12345
# The key must be at this path on the login node AND accessible from compute
# nodes (shared filesystem).
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/leonardo_tunnel}"

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY" >&2
    echo "Generate it on the login node:" >&2
    echo "  ssh-keygen -t ed25519 -f ~/.ssh/leonardo_tunnel" >&2
    echo "  cat ~/.ssh/leonardo_tunnel.pub >> ~/.ssh/authorized_keys" >&2
    exit 1
fi

# --- Start ssh-agent and load key (handles passphrase) ---
# SSH_KEY_PASSPHRASE can be set in secrets.env; defaults to 12345
_PASSPHRASE="${SSH_KEY_PASSPHRASE:-12345}"

eval "$(ssh-agent -s)" >&2

# Use SSH_ASKPASS to feed the passphrase non-interactively
_ASKPASS_SCRIPT="/tmp/.ssh_askpass_${SLURM_JOB_ID:-$$}.sh"
cat > "$_ASKPASS_SCRIPT" <<ASKEOF
#!/bin/sh
echo '$_PASSPHRASE'
ASKEOF
chmod +x "$_ASKPASS_SCRIPT"

DISPLAY=:0 SSH_ASKPASS="$_ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force \
    ssh-add "$SSH_KEY" >&2 2>&1
rm -f "$_ASKPASS_SCRIPT"

echo "SSH key loaded into agent" >&2

# --- Open SSH tunnel (dynamic SOCKS5 port forwarding) ---
NODE_IP=$(nslookup "$NODE_HOST" 2>/dev/null | grep 'Address' | tail -n1 | awk '{print $2}')
if [ -z "$NODE_IP" ]; then
    NODE_IP="$NODE_HOST"
fi

ssh -g -f -N -D "${TUNNEL_PORT}" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=15 \
    -o TCPKeepAlive=no \
    -o ExitOnForwardFailure=yes \
    "${USER}@${LOGIN_NODE}"

echo "SSH tunnel established: localhost:${TUNNEL_PORT} → ${LOGIN_NODE}" >&2

# Save agent env vars for parent process cleanup
_AGENT_ENV="/tmp/.ssh_agent_env_${SLURM_JOB_ID:-$$}"
echo "SSH_AGENT_PID=$SSH_AGENT_PID" > "$_AGENT_ENV"
echo "SSH_AUTH_SOCK=$SSH_AUTH_SOCK" >> "$_AGENT_ENV"
echo "Agent env saved to $_AGENT_ENV" >&2

# --- Generate proxychains config ---
CFG_PATH="${HOME}/.proxychains/proxychains_${SLURM_JOB_ID:-local}.conf"
mkdir -p "$(dirname "$CFG_PATH")"

cat > "$CFG_PATH" <<PCEOF
strict_chain
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF

echo "Proxychains config at $CFG_PATH (socks5://${NODE_IP}:${TUNNEL_PORT})" >&2

# Export for child processes that source this script
export PROXYCHAINS_CONF_FILE="$CFG_PATH"
export PROXYCHAINS_SOCKS5_HOST="${NODE_IP}"
export PROXYCHAINS_SOCKS5_PORT="${TUNNEL_PORT}"
export SSH_AGENT_PID  # from eval ssh-agent above

# The ONLY stdout line — captured by CMD_PREFIX=$(bash this_script.sh)
echo "proxychains4 -q -f $CFG_PATH"
