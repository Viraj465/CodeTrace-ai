# CodeTrace AI demo — render with:  vhs demo.tape   (macOS / Linux / WSL)
# VHS 0.12 captures no frames on native Windows (charmbracelet/vhs#631); there,
# use scripts/record_demo.py + agg instead (see that file's docstring).
# Needs: an Ollama config in ~/.codetrace/config.json and the demo repo indexed
# once beforehand:  codetrace index https://github.com/pallets/itsdangerous

Output Images/demo.gif
Output Images/demo.mp4

Set Shell "pwsh"
Set FontSize 15
Set Width 1280
Set Height 820
Set Padding 24
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 110ms
# Local 4B model answers in ~2 min; 3x keeps the GIF short without cutting anything.
Set PlaybackSpeed 3

Hide
Type "& 'E:\CLI\.venv\Scripts\Activate.ps1'; $env:PYTHONIOENCODING='utf-8'" Enter
Type `Set-Location "$HOME\.codetrace\repos\github.com\pallets\itsdangerous"; Clear-Host` Enter
Sleep 2s
Show

# Re-index: SHA-256 delta sync skips everything that hasn't changed.
Type "codetrace index ." Sleep 600ms Enter
Wait+Screen@120s /Indexing complete|up to date|No changes/
Sleep 4s
Type "Clear-Host" Enter
Sleep 500ms

# Ask a blast-radius question against a local Ollama model.
Type "codetrace chat" Sleep 600ms Enter
Wait+Screen@120s /Session:/
Sleep 3s
Type "What breaks if I change Signer.get_signature?" Sleep 800ms Enter
Sleep 6s
Wait+Line@600s /^You:/
Sleep 15s

Type "exit" Enter
Sleep 2s
