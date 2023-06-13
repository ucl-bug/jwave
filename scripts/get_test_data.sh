# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

echo "Getting test data..."

# Making a temporary directory for test data, remove it if it already exists
if [ -d ".test_data" ]; then
    rm -rf .test_data
fi
mkdir -p .test_data
git clone https://github.com/ucl-bug/jwave-data.git .test_data

# Moving test data to the right place
cp -ur .test_data/tests/* ./tests/ && rm -r .test_data/tests/*

# Removing the temporary directory
rm -rf .test_data

echo "Done."
