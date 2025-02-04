#naive bayes implementation from scratch (for categorical data)

#sample dataset (features and labels)
#each sample has 3 features, and a label (target variable)
data = [
    ['rainy', 'hot', 'high', 'no'],
    ['rainy', 'hot', 'high', 'no'],
    ['overcast', 'hot', 'high', 'yes'],
    ['sunny', 'mild', 'high', 'yes'],
    ['sunny', 'cool', 'normal', 'yes'],
    ['sunny', 'cool', 'normal', 'no'],
    ['overcast', 'cool', 'normal', 'yes'],
    ['rainy', 'mild', 'high', 'no'],
    ['rainy', 'cool', 'normal', 'yes'],
    ['sunny', 'mild', 'normal', 'yes']]

#separate features (X) and labels (Y)
X = [sample[:-1] for sample in data]
Y = [sample[-1] for sample in data]

#step 1: calculate the prior probabilities P(Y)
def calculate_prior(Y):
    class_counts = {}
    for label in Y:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    total = len(Y)
    prior = {label: count / total for label, count in class_counts.items()}
    return prior

#step 2: calculate the likelihood P(X_i | Y)
def calculate_likelihood(X, Y):
    feature_counts = {}
    for i in range(len(X[0])):  #fFor each feature
        feature_counts[i] = {}
        for label in set(Y):
            feature_counts[i][label] = {}
            #collect all feature values for class `label`
            filtered_data = [X[j] for j in range(len(X)) if Y[j] == label]
            feature_values = [sample[i] for sample in filtered_data]
            unique_values = set(feature_values)
            for value in unique_values:
                feature_counts[i][label][value] = feature_values.count(value) / len(feature_values)
    return feature_counts

#step 3: calculate the posterior probability P(Y | X)
def calculate_posterior(X, prior, likelihood):
    posteriors = {}
    for label in prior:
        posterior = prior[label]
        #multiply by the likelihood for each feature given the class
        for i in range(len(X)):
            feature_value = X[i]
            if feature_value in likelihood[i][label]:
                posterior *= likelihood[i][label][feature_value]
            else:
                posterior *= 0  #if feature value was never seen for that class, P(X_i | Y) is 0
        posteriors[label] = posterior
    return posteriors

#step 4: make prediction based on the highest posterior probability
def predict(X_sample, prior, likelihood):
    posteriors = calculate_posterior(X_sample, prior, likelihood)
    return max(posteriors, key=posteriors.get)

#training phase (calculate priors and likelihoods)
prior = calculate_prior(Y)
likelihood = calculate_likelihood(X, Y)

#testing with a new sample
new_sample = ['sunny', 'cool', 'high']  #a new input sample to predict

#predict the class for the new sample
predicted_class = predict(new_sample, prior, likelihood)
print(f"The predicted class for {new_sample} is: {predicted_class}")
