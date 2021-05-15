
### Combined Feature Importance Evaluation

<sub>
- We often are interested in the contribution features have in models or a formulated relation of features to a target variable, for example we might look at the correlation to the target variable of features and remove highly correlated features. The full list of those used in the following function are visible in the figure below.
- Due to the subjective nature of feature importance evaluations of various methods, we may find it useful to visualise them together.
- The following simple code combines the most common approaches and gives them equal weighting and is normalised. Larger scores indicate more contribution in each individual evaluation. 
- The nature of Plotly's interative plots, allows us to compare any combination of methods & visualise the feature importance.
</sub>

|<sub>Example from Kaggle Notebook, [Perth Property Price Prediction](https://www.kaggle.com/shtrausslearning/perth-property-price-prediction)</sub>|
|-|
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejouwc-8bce3c65-0e2f-4707-87e6-ce3f8641d70f.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqb3V3Yy04YmNlM2M2NS0wZTJmLTQ3MDctODdlNi1jZTNmODY0MWQ3MGYucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.XFflHEyeZa7eUuFb1YwSgZQhy2wXZicJUyj11dY7QBA) |
