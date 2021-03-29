# convert to tex
jupyter nbconvert --output-dir 'temp' --to latex $args[0]

# compile to pdf
cd temp
$tex = $args[0].Replace('.ipynb', '.tex')
xelatex $tex -quiet

# delete useless files
cd ..
$pdf = $args[0].Replace('.ipynb', '.pdf')
# delete this file if it exists
if (Test-Path $pdf) {
    Remove-Item $pdf
}
mv temp/$pdf .
rm -r temp