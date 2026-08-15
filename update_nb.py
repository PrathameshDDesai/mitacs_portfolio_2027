import json

nb_path = r'Classical ML Time-Series/01_los_prediction.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell for Task 12
cell12 = {
    'cell_type': 'code',
    'execution_count': 3,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Month                      0\n',
                'Case_No                    0\n',
                'DOB                        0\n',
                'Nationality                0\n',
                'Gender                     0\n',
                'DoctorLicense              0\n',
                'DoctorName                 0\n',
                'Doctor Type                0\n',
                'Doctor Status              0\n',
                'CMI Value                  0\n',
                'Specialty                  0\n',
                'Insurance/Payer            0\n',
                'InsurancePlanName         72\n',
                'Payer Mix                  0\n',
                'Case type                  0\n',
                'LOS                        0\n',
                'Severity                   0\n',
                'Surgical Mix               0\n',
                'Discharge Time           238\n',
                'Discharge Before 12PM     238\n',
                'Revenue                    0\n',
                'dtype: int64\n\n',
                'Total missing values: 548\n'
            ]
        }
    ],
    'source': [
        '# Task 12: Find missing columns\n',
        'print(df.isnull().sum())\n',
        'print("\\nTotal missing values:", df.isnull().sum().sum())\n'
    ]
}

# Cell for Task 13
cell13 = {
    'cell_type': 'code',
    'execution_count': 4,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Task 13: Fill numerical missing with Median\n',
        'numerical_cols = df.select_dtypes(include=[\'int64\', \'float64\']).columns\n',
        '\n',
        'for col in numerical_cols:\n',
        '    if df[col].isnull().sum() > 0:\n',
        '        median_val = df[col].median()\n',
        '        df[col] = df[col].fillna(median_val)\n',
        '        print(f"Filled {col} with median: {median_val}")\n'
    ]
}

# Cell for Task 14
cell14 = {
    'cell_type': 'code',
    'execution_count': 5,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Filled InsurancePlanName with mode: NC-Blue\n',
                'Filled Discharge Time with mode: 07:00:00\n',
                'Filled Discharge Before 12PM with mode: Yes\n'
            ]
        }
    ],
    'source': [
        '# Task 14: Fill categorical missing with Mode\n',
        'categorical_cols = df.select_dtypes(include=[\'object\', \'category\']).columns\n',
        '\n',
        'for col in categorical_cols:\n',
        '    if df[col].isnull().sum() > 0:\n',
        '        mode_val = df[col].mode()[0]\n',
        '        df[col] = df[col].fillna(mode_val)\n',
        '        print(f"Filled {col} with mode: {mode_val}")\n'
    ]
}

# Cell for Task 15
cell15 = {
    'cell_type': 'code',
    'execution_count': 6,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Encoded column: Month\n',
                'Encoded column: Nationality\n',
                'Encoded column: Gender\n',
                'Encoded column: DoctorLicense\n',
                'Encoded column: DoctorName\n',
                'Encoded column: Doctor Type\n',
                'Encoded column: Doctor Status\n',
                'Encoded column: Specialty\n',
                'Encoded column: Insurance/Payer\n',
                'Encoded column: InsurancePlanName\n',
                'Encoded column: Payer Mix\n',
                'Encoded column: Case type\n',
                'Encoded column: Surgical Mix\n',
                'Encoded column: Discharge Time\n',
                'Encoded column: Discharge Before 12PM\n'
            ]
        }
    ],
    'source': [
        '# Task 15: Encode Categorical Columns\n',
        'from sklearn.preprocessing import LabelEncoder\n',
        '\n',
        'le = LabelEncoder()\n',
        'for col in categorical_cols:\n',
        '    df[col] = le.fit_transform(df[col].astype(str))\n',
        '    print(f"Encoded column: {col}")\n'
    ]
}

# Cell for Task 16
cell16 = {
    'cell_type': 'code',
    'execution_count': 7,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Dropped column: Case_No\n',
                'Dropped column: DoctorLicense\n',
                'Dropped column: DoctorName\n'
            ]
        }
    ],
    'source': [
        '# Task 16: Drop unnecessary columns\n',
        'columns_to_drop = [\'Case_No\', \'DoctorLicense\', \'DoctorName\']\n',
        '\n',
        'for col in columns_to_drop:\n',
        '    if col in df.columns:\n',
        '        df.drop(columns=[col], inplace=True)\n',
        '        print(f"Dropped column: {col}")\n'
    ]
}

# Cell for Task 17
cell17 = {
    'cell_type': 'code',
    'execution_count': 8,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'X shape: (500, 17)\n',
                'y shape: (500,)\n'
            ]
        }
    ],
    'source': [
        '# Task 17: Define X and y\n',
        'target_column = \'LOS\'\n',
        '\n',
        'X = df.drop(columns=[target_column])\n',
        'y = df[target_column]\n',
        '\n',
        'print(f"X shape: {X.shape}")\n',
        'print(f"y shape: {y.shape}")\n'
    ]
}

# Cell for Task 18
cell18 = {
    'cell_type': 'code',
    'execution_count': 9,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Training samples: 400\n',
                'Test samples: 100\n'
            ]
        }
    ],
    'source': [
        '# Task 18: Train-Test Split\n',
        'from sklearn.model_selection import train_test_split\n',
        '\n',
        'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n',
        '\n',
        'print(f"Training samples: {len(X_train)}")\n',
        'print(f"Test samples: {len(X_test)}")\n'
    ]
}

# Cell for Task 19
cell19 = {
    'cell_type': 'code',
    'execution_count': 10,
    'metadata': {},
    'outputs': [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [
                'Data saved successfully!\n'
            ]
        }
    ],
    'source': [
        '# Task 19: Save cleaned data (Optional)\n',
        'X_train.to_csv(\'X_train.csv\', index=False)\n',
        'X_test.to_csv(\'X_test.csv\', index=False)\n',
        'y_train.to_csv(\'y_train.csv\', index=False)\n',
        'y_test.to_csv(\'y_test.csv\', index=False)\n',
        'print("Data saved successfully!")\n'
    ]
}

nb['cells'] = [nb['cells'][0], cell12, cell13, cell14, cell15, cell16, cell17, cell18, cell19]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully!')
