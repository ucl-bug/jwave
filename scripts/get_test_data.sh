echo "Getting test data..."

# Making a temporary directory for test data, remove it if it already exists
if [ -d ".test_data" ]; then
    rm -rf .test_data
fi
mkdir -p .test_data
git clone https://github.com/ucl-bug/jwave-data.git .test_data

# Moving test data to the right place
mv .test_data/tests/kwave_data/* ./tests/kwave_data

# Removing the temporary directory
rm -rf .test_data

echo "Done."
