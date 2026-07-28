import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

for name in ['kidney', 'liver', 'parkinsons']:
    path = f'../Project_4/best_{name}_model.pkl'
    try:
        with open(path, 'rb') as f:
            p = pickle.load(f)
        print(f'\n--- {name.upper()} ---')
        print('Features:', p.get('feature_names', []))
        if 'encoders' in p:
            print('Encoders:', list(p['encoders'].keys()))
            for col, enc in p['encoders'].items():
                print(f"  {col} classes: {list(enc.classes_)}")
    except Exception as e:
        print(f'{name} failed:', type(e).__name__, e)
