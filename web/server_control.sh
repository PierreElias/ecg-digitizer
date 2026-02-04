#!/bin/bash
# ECG Digitizer Server Control Script
# Usage: ./server_control.sh [start|stop|restart|status|logs]

PLIST=~/Library/LaunchAgents/com.ecg.digitizer.server.plist
USER_ID=$(id -u)

case "$1" in
    start)
        echo "Starting ECG server..."
        launchctl bootstrap gui/$USER_ID $PLIST 2>/dev/null || launchctl kickstart gui/$USER_ID/com.ecg.digitizer.server
        sleep 2
        curl -s http://localhost:8080/ > /dev/null && echo "✅ Server started on http://localhost:8080" || echo "❌ Failed to start"
        ;;
    stop)
        echo "Stopping ECG server..."
        launchctl bootout gui/$USER_ID/com.ecg.digitizer.server 2>/dev/null
        echo "✅ Server stopped"
        ;;
    restart)
        echo "Restarting ECG server..."
        launchctl kickstart -k gui/$USER_ID/com.ecg.digitizer.server
        sleep 2
        curl -s http://localhost:8080/ > /dev/null && echo "✅ Server restarted" || echo "❌ Failed to restart"
        ;;
    status)
        echo "ECG Server Status:"
        echo "=================="
        if curl -s http://localhost:8080/ > /dev/null 2>&1; then
            echo "✅ Server is running on http://localhost:8080"
            ps aux | grep -v grep | grep "python.*app.py" | awk '{print "   PID: "$2"  Memory: "$4"%"}'
        else
            echo "❌ Server is not responding"
        fi
        ;;
    logs)
        echo "=== Server Logs ==="
        tail -50 /tmp/ecg_server.log 2>/dev/null
        echo ""
        echo "=== Error Logs ==="
        tail -20 /tmp/ecg_server_error.log 2>/dev/null
        ;;
    *)
        echo "ECG Digitizer Server Control"
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the server"
        echo "  stop    - Stop the server"
        echo "  restart - Restart the server"
        echo "  status  - Check server status"
        echo "  logs    - View server logs"
        ;;
esac
