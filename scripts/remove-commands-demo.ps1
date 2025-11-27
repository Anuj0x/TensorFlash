# Demonstration: Individual Remove-Item commands vs Combined command

Write-Host "=== INDIVIDUAL Remove-Item COMMANDS (separate for each file type) ===" -ForegroundColor Yellow

# Individual Remove-Item commands for each file type
Write-Host "Remove-Item -Path '*.so' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.pyd' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.dll' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.lib' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.exp' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.pdb' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.tmp' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.temp' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.log' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.cache' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.bak' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.swp' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.swo' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*~' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.orig' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.rej' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.old' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.o' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.obj' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.ptx' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.cubin' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '*.fatbin' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path 'Thumbs.db' -Force" -ForegroundColor Red
Write-Host "Remove-Item -Path '.DS_Store' -Force" -ForegroundColor Red

Write-Host "`n=== COMBINED COMMAND (all file types in one operation) ===" -ForegroundColor Green

# Combined command using pipeline
Write-Host '"*.so", "*.pyd", "*.dll", "*.lib", "*.exp", "*.pdb", "*.tmp", "*.temp", "*.log", "*.cache", "*.bak", "*.swp", "*.swo", "*~", "*.orig", "*.rej", "*.old", "*.o", "*.obj", "*.ptx", "*.cubin", "*.fatbin", "Thumbs.db", ".DS_Store" | ForEach-Object { Get-ChildItem -Path . -Include $_ -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -Verbose }' -ForegroundColor Green

Write-Host "`n=== EXECUTING THE COMBINED COMMAND ===" -ForegroundColor Cyan

# Execute the combined command
"*.so", "*.pyd", "*.dll", "*.lib", "*.exp", "*.pdb", "*.tmp", "*.temp", "*.log", "*.cache", "*.bak", "*.swp", "*.swo", "*~", "*.orig", "*.rej", "*.old", "*.o", "*.obj", "*.ptx", "*.cubin", "*.fatbin", "Thumbs.db", ".DS_Store" | ForEach-Object {
    Get-ChildItem -Path . -Include $_ -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -Verbose
}

Write-Host "`n✅ Combined command executed successfully!" -ForegroundColor Green
