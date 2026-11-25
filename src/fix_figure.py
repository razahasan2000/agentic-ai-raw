with open('clean_pipeline_final.py', 'r') as f:
    content = f.read()

# Find the figure generation section
idx = content.find('# LEFT COLUMN: DETERMINISTIC TOOLS')
if idx >= 0:
    # Find the end of the figure generation section
    end_marker = 'plt.savefig(RESULTS_DIR'
    end_idx = content.find(end_marker, idx)
    if end_idx >= 0:
        # Find the end of that line
        end_idx = content.find('\n', end_idx) + 1
        # Find plt.close() and print statement
        close_idx = content.find('plt.close()', end_idx)
        if close_idx >= 0:
            close_idx = content.find('\n', close_idx) + 1
            print_idx = content.find('print("   Generated fig_architecture.png")', close_idx)
            if print_idx >= 0:
                end_idx = content.find('\n', print_idx) + 1
            else:
                end_idx = close_idx
        else:
            end_idx = len(content)
    else:
        end_idx = len(content)
    
    print('Section from', idx, 'to', end_idx, 'length:', end_idx - idx)
    
    # Read the new figure code
    with open('new_figure_code.py', 'r') as f:
        new_figure_code = f.read()
    
    # Replace the section
    new_content = content[:idx] + new_figure_code + content[end_idx:]
    
    with open('clean_pipeline_final.py', 'w') as f:
        f.write(new_content)
    
    print('Figure generation code replaced successfully')
else:
    print('Section not found')