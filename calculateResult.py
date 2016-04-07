import sys
filename = sys.argv[1]
movieSummary = open(filename, 'rb')
positive = 0
negative = 0
neutral = 0
for summary in movieSummary:
	#print comment
	words = summary.split(':')
	if summary.find('Positive') != -1:
		positive += int(words[1])
	elif summary.find('Negative') != -1:
		negative += int(words[1])
	elif summary.find('Neutral') != -1:
		neutral += int(words[1])
print "Positive:", positive, "Negative:", negative, "Neutral",neutral
print "Positive ratio:" , float(positive)/float(positive+negative+neutral)
print "PT/NT:", float(positive)/float(negative)