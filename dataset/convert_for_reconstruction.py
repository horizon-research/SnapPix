import csv
import argparse

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile, delimiter=' ')

        for row in reader:
            if len(row) == 2:  # Ensure there are two columns in the row
                new_row = [row[0], '0', '-1']
                writer.writerow(new_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV input and output paths.")
    parser.add_argument("input_path", help="Path to the input CSV file.")
    parser.add_argument("output_path", help="Path to the output CSV file.")

    args = parser.parse_args()
    process_csv(args.input_path, args.output_path)
