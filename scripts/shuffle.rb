#!/usr/bin/ruby

require 'fileutils'

def read_file(filepath)
    line_num=0
    text=File.open(filepath).read
    text.gsub!(/\r\n?/, "\n")
    lines = []
    text.each_line do |line|
        lines << line
    end
    lines
end


def write_file(filepath, lines)
    file = File.new(filepath, "w")
    lines.each do |line|
        file.puts line
    end
    file.close
end

filepath = "/home/gustavoneves/data/gemini/jequitaia/orientations2/orientations/01_source/source.valid.list"
list = read_file(filepath)
write_file(filepath, list.shuffle)
