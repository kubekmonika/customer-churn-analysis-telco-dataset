import sys
import srs


def main():
    if len(sys.argv) == 3:
        data_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\nDATASET: {data_filepath}')
        data = srs.load_data(data_filepath)

        print('Cleaning data...')
        data = srs.clean_data(data)

        print('Transforming data...')
        data = srs.transform_data(data)

        print(f'Saving data...\nDATABASE: {database_filepath}')
        srs.save_data(data, database_filepath)

        print('Done!')
    else:
        print(
            'Please provide the filepaths of the data and the database '\
            '\nExample: '\
            'python process_data.py '\
            'WA_Fn-UseC_-Telco-Customer-Churn.csv '\
            'customers.db'
        )


if __name__ == '__main__':
    main()
