import sys
import srs


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        model_filepath = model_filepath.replace('.pkl', '')

        print(f'Loading data...\nDATABASE: {database_filepath}')
        data = srs.load_data(database_filepath)

        print('Building a model...')
        model = srs.build_model(data)
        print(model)

        print('Evaluating model...')
        srs.evaluate_model(model)

        print(f'Saving model...\nMODEL: {model_filepath}.pkl')
        srs.save_model_(model, model_filepath)

        print('Done!')
    else:
        print(
            'Please provide the filepath of the database '\
            'and the filepath of the model'\
            '\nExample: '\
            'python train_classifier.py '\
            'customers.db '\
            'model.pkl'
          )


if __name__ == '__main__':
    main()
