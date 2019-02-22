for degree in $(seq 1 5); do
for sigma in "0.001" "0.005" "0.1" "0.3" "0.5" "0.8" "1" "2" "3" "5" "10"; do
    python tuning.py --sigma ${sigma} --degree ${degree} --dataset pubmed
done; done
