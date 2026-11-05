# Open the report in Microsoft Word, refresh the contents table and page fields, save, and export a PDF.
$ErrorActionPreference = "Stop"
$docx = Join-Path $PSScriptRoot "M2_DeepLabV3Plus_Results_Report.docx"
$pdf  = [IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docx)
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $doc.SaveAs2($pdf, 17)   # 17 = wdFormatPDF
    $doc.Close()
    Write-Output "wrote $pdf"
} finally {
    $word.Quit()
}
