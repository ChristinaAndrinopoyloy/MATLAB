function [total_accuracy,total_sensitivity,total_specificity] = final_results(accuracy,sensitivity,specificity,repets)
total_accuracy = accuracy / repets;
total_sensitivity = sensitivity / repets;
total_specificity = specificity / repets;
end

