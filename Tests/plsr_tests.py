
-- UNIT TESTS ---

def test_plsr_q2y_scores_are_valid(X_train_scaled, Y_train):
    # tests if Q2Y scores are in valid range
    for n in range(1, 5):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_scaled, Y_train, cv=5, scoring='r2')
        assert scores.min() >= -1.0, \
            f"Q2Y score below -1: {scores.min()}"
        assert scores.max() <= 1.0, \
            f"Q2Y score above 1: {scores.max()}"

def test_plsr_q2y_improves_with_components(X_train_scaled, Y_train):
    # tests that Q2Y improves as number of components increases up to 4
    q2y = []
    for n in range(1, 5):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_scaled, Y_train, cv=5, scoring='r2')
        q2y.append(scores.mean())
    
    assert q2y[1] > q2y[0], \
        f"Q2Y should improve from component 1 to 2, got {q2y[0]:.4f} and {q2y[1]:.4f}"
    assert q2y[2] > q2y[1], \
        f"Q2Y should improve from component 2 to 3, got {q2y[1]:.4f} and {q2y[2]:.4f}"
    assert q2y[3] > q2y[2], \
        f"Q2Y should improve from component 3 to 4, got {q2y[2]:.4f} and {q2y[3]:.4f}"

def test_diabetes_loading_decreases_across_components(X_train_scaled, Y_train):
    # tests that diabetes loading decreases with each component
    # meaning each component contributes less to predicting diabetes
    pls = PLSRegression(n_components=4)
    pls.fit(X_train_scaled, Y_train)
    
    diabetes_loadings = abs(pls.y_loadings_[0])
    
    assert diabetes_loadings[0] > diabetes_loadings[1], \
        f"Diabetes loading should decrease from component 1 to 2, got {diabetes_loadings[0]:.4f} and {diabetes_loadings[1]:.4f}"
    assert diabetes_loadings[1] > diabetes_loadings[2], \
        f"Diabetes loading should decrease from component 2 to 3, got {diabetes_loadings[1]:.4f} and {diabetes_loadings[2]:.4f}"
    assert diabetes_loadings[2] > diabetes_loadings[3], \
        f"Diabetes loading should decrease from component 3 to 4, got {diabetes_loadings[2]:.4f} and {diabetes_loadings[3]:.4f}"
