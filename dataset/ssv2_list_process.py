import argparse
import csv
import os

def process_csv(input_path, output_path, pretrained=False):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile)
        
        # Process each row in the input CSV
        for row in reader:
            path = row[0]
            # Extract the frame number and last number
            frame_number = os.path.basename(path)
            last_number = row[-1]
            # Prepare the output format
            output_path = f"ssv2_processed/{frame_number}.mp4"
            if pretrained:
                writer.writerow([output_path + " 0 -1"])
            else:
                writer.writerow([output_path + " " + last_number])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input.csv to create output.csv.")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("output", help="Path to output CSV file")
    # add a store true called pretrained
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model for training")
    args = parser.parse_args()
    
    process_csv(args.input, args.output, args.pretrained)
