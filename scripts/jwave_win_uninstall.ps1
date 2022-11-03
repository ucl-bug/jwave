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

Write-Output "-- Uninstalling jwave"

# Remove scoop
scoop uninstall scoop

# Remove jwave folder if exists
Remove-Item -Path "~/.jwave" -Force -Recurse -ErrorAction SilentlyContinue

Write-Output "-- jwave uninstalled"