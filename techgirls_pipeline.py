import pandas as pd
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


addresses = pd.read_csv('data/misc/addresses.csv')
train = pd.read_csv('data/train/train.csv')
sample = pd.read_csv('data/sample_submission.csv', sep=';')

shipments1 = pd.read_csv('data/shipments/shipments2020-01-01.csv')
shipments2 = pd.read_csv('data/shipments/shipments2020-03-01.csv')
shipments3 = pd.read_csv('data/shipments/shipments2020-04-30.csv')
shipments4 = pd.read_csv('data/shipments/shipments2020-06-29.csv')

shipments = pd.concat([shipments1, shipments2, shipments3, shipments4])

train['uid'] = train.phone_id.astype(str) + '_'+ train.order_completed_at.str[5:]

shipments.order_completed_at = pd.to_datetime(shipments.order_completed_at)

shipments = shipments.merge(addresses, how='left', left_on='ship_address_id', right_on='id')
shipments = shipments.merge(train, how='left', on='phone_id')

shipments = shipments.rename(columns={'order_completed_at_x': 'order_month',
                          'order_completed_at_y': 'last_order_month_from_train'})

shipments['month_order'] = shipments.order_month.dt.month
shipments['last_order_month'] = shipments.last_order_month_from_train.str[6:]

shipments = shipments[shipments.last_order_month.notna()]

shipments['month_order'] = shipments['month_order'].apply(lambda x: 1 if x == 12 else x)

shipments = shipments[(shipments.last_order_month.astype(int) >= shipments.month_order) & 
                     ((shipments.last_order_month.astype(int) - shipments.month_order == 1))]

train.loc[train['order_completed_at'].str[6:] == '7', 'is_for_test'] = 1
train.loc[train['order_completed_at'].str[6:] != '7', 'is_for_test'] = 0

train['mean_weight'] = train['uid'].map(shipments.groupby(['uid']).total_weight.mean())

shipments['retailer_churn_rate'] = shipments.retailer.map(shipments[shipments.last_order_month != 7].groupby('retailer').target.mean())
train = train.merge(shipments[['uid', 'retailer_churn_rate']], how='left', on='uid')

a_df = pd.read_csv('features_shipment_line_items_without_null.csv')
a_df.phone_id = a_df.phone_id.astype(int)
a_df['uid'] = a_df.phone_id.astype(str) + '_'+ a_df.month_order_completed_tr.str[5:]

train = train.merge(a_df, how='left', on='uid')
train = train.drop_duplicates('uid')
features = [
    'mean_weight',
    'retailer_churn_rate',
    'shipment_id', 
    'total_receipt',
    'total_cost', 
    'discount', 
    'replaced', 
    'cancelled', 
    'shipped_item_count',
    'promo_total', 
    'platform', 
    'os', 
    'dw_kind', 
    'ship_address_id',
    'shipment_state', 
    'time_delivery_h', 
    'time_shipment_h'
]

X = train[(train.is_for_test == 0)][features]
y = train[(train.is_for_test == 0)]['target']
y_valid_pred = 0*y
y_test_pred = 0
X_test = train[train.is_for_test == 1][features]
y_test = train[train.is_for_test == 1]['target']

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)

model = CatBoostClassifier(
    learning_rate=0.06, 
    depth=6, 
    l2_leaf_reg = 14, 
    iterations = 650,
    verbose = False,
    loss_function='Logloss'
)

X = train[(train.is_for_test == 0)][features]
y = train[(train.is_for_test == 0)]['target']
X_test = train[train.is_for_test == 1][features]
y_test = train[train.is_for_test == 1]['target']

y_valid_pred = 0*y
y_test_pred = 0

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", i)
    
    # Run model for this fold

    fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
#     print( "  f1 = ", f1_score(y_valid, pred))
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    
y_test_pred /= K  # Average test set predictions

thr1 = 0.4
f1_score(y_test, [1 if x>= thr1 else 0 for x in y_test_pred])

y_test_pred_class = [1 if x>= thr1 else 0 for x in y_test_pred]

test = train[train.is_for_test == 1]
test['pred'] = y_test_pred_class
test.pred.value_counts()

test = test[['phone_id_x', 'pred']].drop_duplicates()
sample = sample.merge(test[['phone_id_x', 'pred']], how='left', left_on='Id', right_on = 'phone_id_x')
sample = sample.drop(columns=['Predicted', 'phone_id_x']).rename(columns={'pred':'Predicted'})

sample.Predicted = sample.Predicted.fillna(1).astype(int)
true_sample = pd.read_csv('data/sample_submission.csv', sep=';')
sample[sample['Id'].isin(true_sample.Id.values)].drop_duplicates('Id').to_csv('sample_submission5.csv', index=False)