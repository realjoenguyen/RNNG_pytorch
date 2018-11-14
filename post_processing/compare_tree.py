
pred_seqs_file = 'pred_dev_seq.txt'
test_seqs_file = 'dev_seqs.txt'
from postprocess import get_diff_prods_no_span
diff_no_span, diff_counter, diff_height_counter = get_diff_prods_no_span(test_seqs_file, pred_seqs_file)
print ('len diff =', len(diff_no_span))
print (diff_counter.most_common(10))
print ('')
print(diff_height_counter.most_common(10))