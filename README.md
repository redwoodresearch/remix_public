# REMIX Participant Instructions

*See [here](#first-time-setup) for first time setup.*

## Course Outline

- W0D1 - pre-course exercises on PyTorch and einops (CPU)
- D1 - Intro to Circuits (CPU)
- D2 - Build your own GPT in Circuits (CPU)
- D3 - Induction Heads Investigation (GPU preferred)
- D4a - Intro to Causal Scrubbing (GPU preferred)
- D4b - Paren Balencing (GPU preferred)
- D5 - Indirect Object Identification and Causal Scrubbing (GPU preferred)

## Every day instructions

### Connect to your GPU instance

1. Get your pair's instance IP address from the pairs sheet bookmarked in the `#1-general` channel in the REMIX slack.
2. Set the IP address in your ssh config file: [open the config file](#3-Set-up-your-ssh-config-file), edit the `HostName` line with the IP address for the day, and save the file.
3. Connect to the instance
	- Click the ![](https://i.imgur.com/K3gXTqk.png) button in the bottom left, choose "Connect to Host...", and choose "remix"
	- Click the file button in the top left (looks like ![](https://i.imgur.com/hP9aTIh.png=20x20)), click "Open Folder", choose the "mlab2" directory, and hit "OK"
	- If it asks you if you trust the authors, click "Yes"
	
*If this is your first time connecting, make sure the `python` extension is installed in VSCode via the extensions menu (icon looks like ![](https://i.imgur.com/2loWEul.png=20x20)).*

### Using git

The REMIX instructions are all stored in this branch (`remix`) of the `mlab2` repository. Each day you should make a new branch off of `remix`, make commits to it and push it up to the repo. This is how your work during MLAB will be saved for you to reference later.

It's important to `git pull` at the start of each day. Then make a branch for the day: `git checkout -b <branch name> remix`, where your branch name should follow the convention `remix_d#/<name>-and-<name>`.

For example, if Tamera and Edmund were pairing on the day 3 content, the command would be `git checkout -b remix_d3/tamera-and-edmund remix`
If you share a first name with someone else in the program, you should probably include your last initial or last name in your branch name for disambiguation. Create a new file for your answers and work through the material with your partner.

To view the instructions for a day, right-click the `remix_d#_instructions.md` file in the file menu on the left, and click "Open Preview". 
You'll be writing all your code each day in a new file that you'll create. We recommend making a new `.py` file (suggested name: `remix_d#_answers.py`) and starting it with the characters `# %%`.

Then paste in a block of code from the instructions doc and add another `# %%` line. In VSCode, if you have the Python extension installed, you should see the option to "Run Cell" at the top of the file; this will start an interactive session you can use to run the code. If you add more code at the bottom of the file and follow it with another `# %%`, this will create another cell which can be run independently in the same session. Cells can be run many times and in any order you choose; the session will maintain variables and state until it is restarted. Add code cells and run them, filling in your answers as you go, to carry out the exercises. Select the `remix` kernel by clicking the `select kernel` button in the top right corner of the interactive panel.

As you work, commit changes to your branch and push them to the repo.

To make a commit:

```bash
git add :/
git commit -m '<your commit message>'
```

To push your commits:

- The first time pushing your branch: `git push -u origin <branch name>` (or run git push and copy and paste the suggested command)
- Every other time: `git push`
- Make sure to commit and push at least once at the end of the day! We'll reset the repo on all instances each night, so if you haven't pushed your work to GitHub it will be lost.

## First time setup

### 1. Download the `remix_ssh` key and put it in your ssh directory
1. Download the `remix_ssh` key file, which is pinned in the `#1-general` channel in the Slack.
2. Make an ssh directory (if you don't already have one).
	- Linux / mac command: `mkdir -p ~/.ssh`
	- Windows command: `md -Force ~\.ssh`
		-  If that doesn't work, try `md -Force C:\Users\[your user name]\.ssh` 
		-  If that doesn't work for you, confirm that this folder already exists for yourself, and if not make it in the file navigator application.
3. Copy the downloaded `remix_ssh` key to your ssh directory.
	- All platforms command (assumes the key is at `~/Downloads/remix_ssh`):
	  `cp ~/Downloads/remix_ssh ~/.ssh/remix_ssh`
4. Set permissions on the key.
	- Linux / mac command: `chmod 600 ~/.ssh/remix_ssh`
	- Windows: I haven't tested it myself but [these instructions](https://superuser.com/a/1296046) seem like the they might do the trick ¯\\\_(ツ)_/¯

If you still see `permission denied` errors, make sure that your `remix_ssh` file is using LF line endings and has a final line break at the end (so there shouldn't be any text on the last line of the file).

### 2. Install the VSCode remote-ssh extension
1. Have [VSCode](https://code.visualstudio.com/) installed.
2. Open VSCode. In the vertical toolbar on the very left, click on the extensions menu (icon looks like ![](https://i.imgur.com/2loWEul.png=20x20)).
3. Search for "remote ssh". This first result is the one you want to install; click the blue "Install" button.
   ![](https://i.imgur.com/RkQsGzy.png)


### 3. Set up your ssh config file
1. Click the ![](https://i.imgur.com/K3gXTqk.png) button in the bottom left of VSCode, which brings up the remote ssh options menu.
2. Choose "Open SSH Configuration File...", and if it prompts you to choose a file pick the first one.
3. Paste in this block and then save the file. (If there's already stuff in the file, put this at the end.)
   ```
   Host remix
     HostName <instance ip address>
     User ubuntu
     IdentityFile ~/.ssh/remix_ssh
     StrictHostKeyChecking no
   ```
   If you have an IP address for your pair's GPU instance, replace `<instance ip address>` with that IP address.
   
*Then proceed to the [every day instructions](#every-day-instructions).*
