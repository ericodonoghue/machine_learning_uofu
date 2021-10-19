file="$1"
if [file = 0]
  then
  python3 eo_AdaBoost.py
elif [file = 1]
  then
  python3 eo_BaggedTrees.py
elif [file = 2]
  then
  python3 eo_RandomForests.py
fi