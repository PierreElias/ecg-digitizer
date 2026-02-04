#!/usr/bin/env ruby
require 'xcodeproj'

project_path = 'ECGDigitizer.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Find the target
target = project.targets.find { |t| t.name == 'ECGDigitizer' }

# Navigate to the Results group
main_group = project.main_group
ecg_group = main_group['ECGDigitizer']
presentation_group = ecg_group['Presentation']
flows_group = presentation_group['Flows']
results_group = flows_group['Results']

# Create file reference if it doesn't exist
file_path = 'ECGDigitizer/Presentation/Flows/Results/DiagnosticUploadView.swift'
existing_file = results_group.files.find { |f| f.path == 'DiagnosticUploadView.swift' }

if existing_file
  puts "File already exists in project"
else
  # Add the file reference
  file_ref = results_group.new_reference('DiagnosticUploadView.swift')

  # Add to build phase
  target.source_build_phase.add_file_reference(file_ref)

  # Save
  project.save

  puts "Successfully added DiagnosticUploadView.swift to Xcode project!"
end
