function [temp_accuracy,temp_sensitivity,temp_specificity] = prediction_reality(lbl,tst_ans)
   % check prediction and reality
   true_positive = 0; true_negative = 0; false_positive = 0; false_negative = 0;
    [row, column] = size(lbl);
    for k = 1:row
        if lbl(k) == tst_ans(k)
            if lbl(k) == 1
                true_positive = true_positive + 1;
            else
                true_negative = true_negative + 1;
            end    
        else
            if lbl(k) == 1
                false_negative = false_negative + 1;
            else
                false_positive = false_positive + 1;
            end    
        end
    end 
    
    temp_accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative);
    temp_sensitivity = true_positive/(true_positive+false_negative);
    temp_specificity = true_negative/(true_negative+false_positive);
end

