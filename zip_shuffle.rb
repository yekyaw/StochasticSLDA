def main
  file1 = ARGV[0]
  file2 = ARGV[1]
  lines1 = IO.readlines(file1)
  lines2 = IO.readlines(file2)
  File.open(file1, 'w') do |f1|
    File.open(file2, 'w') do |f2|
      lines1.zip(lines2).shuffle.each do |line1, line2|
        f1.write(line1)
        f2.write(line2)
      end
    end
  end
end

main
