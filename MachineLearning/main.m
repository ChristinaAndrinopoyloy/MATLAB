data = xlsread('./breast-cancer-wisconsin.xlsx','','','basic');  % read data from file
data(:,1) = []; % remove id
famous = mode(data); % famous of column

% fill nan cells
[row, column] = size(data);
for i = 1:column
     for k = 1:row
         if isnan(data(k,i))
             data(k,i) = famous(i);
         end
     end
end

% normalize
rng('default');
data_min = repmat(min(data),row,1);
data_max = repmat(max(data),row,1);
normalized_data = (data-data_min)./(data_max-data_min);

% remove answers from data
answer = normalized_data(:,column);
normalized_data(:,column) = [];
[row, column] = size(normalized_data);
% initialization
temp_accuracy = 0; temp_sensitivity = 0; temp_specificity = 0;
perceptron_accuracy = 0; perceptron_sensitivity = 0; perceptron_specificity = 0;
knn_accuracy = 0; knn_sensitivity = 0; knn_specificity = 0;
bayes_accuracy = 0; bayes_sensitivity = 0; bayes_specificity = 0;
svm_accuracy = 0; svm_sensitivity = 0; svm_specificity = 0;
dtree_accuracy = 0; dtree_sensitivity = 0; dtree_specificity = 0;
% 10-fold cross-validation
indices = crossvalind('Kfold',answer,10);
for i = 1:10
    data_test = (indices == i); 
    data_train = ~data_test;
    
    tst_data = normalized_data(data_test,:);
    tst_ans = answer(data_test,:);
    trn_data = normalized_data(data_train,:);
    trn_ans = answer(data_train,:);
    
    % multilayer perceptron
    net = fitnet(10, 'hiddenSizes', [10 5 3 2 1], 'trainFcn', 'trainbr');
    net = train(net,trn_data',trn_ans');
    lbl = net(tst_data');
    temp = lbl';
    sz = size(lbl');
    myzero = zeros(sz);
    myone = ones(sz);
    distance_from_zero = zeros(sz);
    distance_from_one = zeros(sz);
    for j = 1:sz(1)
        distance_from_zero(j) = sqrt(sum(sum((myzero(j)-temp(j)).*(myzero-temp(j)))));
        distance_from_one(j) = sqrt(sum(sum((myone-temp(j)).*(myone-temp(j)))));    
        if distance_from_zero(j) > distance_from_one(j)
            lbl(j) = 1;
        else
            lbl(j) = 0;
        end    
    end        
    [temp_accuracy, temp_sensitivity,temp_specificity] = prediction_reality(lbl',tst_ans);
    perceptron_accuracy = perceptron_accuracy + temp_accuracy;
    perceptron_sensitivity = perceptron_sensitivity + temp_sensitivity;
    perceptron_specificity = perceptron_specificity + temp_specificity;
    
    %K-Nearest Neighbor
    Mdl = fitcknn(trn_data,trn_ans,'NumNeighbors',10,'Standardize',1,'Distance','cosine');
    lbl = predict(Mdl,tst_data);
    [temp_accuracy, temp_sensitivity,temp_specificity] = prediction_reality(lbl,tst_ans);
    knn_accuracy = knn_accuracy + temp_accuracy;
    knn_sensitivity = knn_sensitivity + temp_sensitivity;
    knn_specificity = knn_specificity + temp_specificity;
    
    % Bayes
    Mdl = fitcnb(trn_data,trn_ans, 'DistributionNames','kernel','Prior','uniform');
    lbl = predict(Mdl,tst_data);
    [temp_accuracy, temp_sensitivity,temp_specificity] = prediction_reality(lbl,tst_ans);
    bayes_accuracy = bayes_accuracy + temp_accuracy;
    bayes_sensitivity = bayes_sensitivity + temp_sensitivity;
    bayes_specificity = bayes_specificity + temp_specificity;
    
    %SVM
    Mdl = fitcsvm(trn_data,trn_ans, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
    lbl = predict(Mdl,tst_data);
    [temp_accuracy, temp_sensitivity,temp_specificity] = prediction_reality(lbl,tst_ans);
    svm_accuracy = svm_accuracy + temp_accuracy;
    svm_sensitivity = svm_sensitivity + temp_sensitivity;
    svm_specificity = svm_specificity + temp_specificity;
    
    % decision tree
    Mdl = fitctree(trn_data,trn_ans, 'MaxNumCategories', 5, 'MergeLeaves', 'off');
    lbl = predict(Mdl,tst_data);
    [temp_accuracy, temp_sensitivity,temp_specificity] = prediction_reality(lbl,tst_ans);
    dtree_accuracy = dtree_accuracy + temp_accuracy;
    dtree_sensitivity = dtree_sensitivity + temp_sensitivity;
    dtree_specificity = dtree_specificity + temp_specificity;

end
[total_perceptron_accuracy,total_perceptron_sensitivity,total_perceptron_specificity] = final_results(perceptron_accuracy,perceptron_sensitivity,perceptron_specificity, 10)
[total_knn_accuracy,total_knn_sensitivity,total_knn_specificity] = final_results(knn_accuracy,knn_sensitivity,knn_specificity, 10)
[total_bayes_accuracy,total_bayes_sensitivity,total_bayes_specificity] = final_results(bayes_accuracy,bayes_sensitivity,bayes_specificity, 10)
[total_svm_accuracy,total_svm_sensitivity,total_svm_specificity] = final_results(svm_accuracy,svm_sensitivity,svm_specificity, 10)
[total_dtree_accuracy,total_dtree_sensitivity,total_dtree_specificity] = final_results(dtree_accuracy,dtree_sensitivity,dtree_specificity, 10)

