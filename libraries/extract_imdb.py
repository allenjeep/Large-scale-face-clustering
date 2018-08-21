import scipy.io
import numpy as np
import cv2
import os
import random

name_column = 4
file_column = 2
coordinate_column = 5
faceprob_column = 6

def save_image(src, dst, coordinates):
	image = cv2.imread(src)
	(x,y,w,h) = np.array(coordinates).astype(int)
	face = image[y:y + h, x:x + w]
	cv2.imwrite(dst, face)

def generate_test_set(imdb, percentage, person, input_dir, output_dir):
	print("Generating test set for " + person + " (" + str(percentage) + "%)")
	rows = len(imdb[0][0])
	
	test_dir = output_dir + '/' + str(percentage)
	
	if not os.path.isdir(test_dir):
		os.mkdir(test_dir)
		
	count = 0
	for row in range(rows):
		if count >= percentage:
			break
			
		if imdb[name_column][0][row][0] != person:
			continue

		if str(imdb[faceprob_column][0][row]) == "-inf":
			continue
		try:
			save_image(input_dir + '/' + imdb[file_column][0][row][0],
						test_dir + '/' + imdb[name_column][0][row][0] + "." +  str(row) + ".jpg",
						imdb[coordinate_column][0][row][0])
			count += 1
		except:
			error = 1
			
	while count < 100:
		row = random.randint(0,rows-1)
		
		if str(imdb[faceprob_column][0][row]) == "-inf":
			continue
		try:
			save_image(input_dir + '/' + imdb[file_column][0][row][0],
						test_dir + '/' + imdb[name_column][0][row][0] + "." +  str(row) + ".jpg",
						imdb[coordinate_column][0][row][0])
			count += 1
		except:
			error = 1
			
	print("Generated test set with " + str(percentage) + "% largest cluster");
	

def generate_test_sets(mat_file, input_dir, output_dir):
	# We want 9 test sets
	# Each set contains X% images of one person and 100-X% of random other people
	# X is 10, 20, 30, 40, 50, 60, 70, 80, 90
	# Each test features one of the 9 most frequent people in the Imdb set
	# Each test contains 100 images
	print("Generating test sets...")
	
	if not os.path.isdir(input_dir):
		print("Could not find folder " + input_dir)
		return False
	
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	
	print("Loading imdb metadata...")
	imdb = scipy.io.loadmat(mat_file)['imdb'][0][0]
	
	people = ['Zooey Deschanel', 
				'Neil Patrick Harris', 
				'Nicole Kidman', 
				'Robert Downey Jr.', 
				'Tom Cruise', 
				'Courteney Cox', 
				'Angelina Jolie', 
				'Brad Pitt', 
				'Jennifer Aniston']
	
	for i in range(len(people)):
		percentage = (i + 1) * 10
		person = people[i]
		
		generate_test_set(imdb, percentage, person, input_dir, output_dir)	
	

def generate_training_set(mat_file, input_dir, output_dir):
	imdb = scipy.io.loadmat(mat_file)['imdb'][0][0]
	rows = len(imdb[0][0])
	
	if not os.path.isdir(input_dir):
		print("Could not find folder " + input_dir)
		return False
	
	if not os.path.isdir(output_dir):
		print("Could not find img output folder... creating...")
		os.mkdir(output_dir)
	
	people = ['Aidan Turner', 'Alexis Denisof']
	
	#people = [
	# 'Ryan Rottman','Sad Taghmaoui','Sami Gayle','Sanjaya Malakar','Scott Eastwood','Sharon Lawrence','Shaun Toub','Skai Jackson','Sophia Myles','Sting','T Bone Burnett','Tammy Blanchard','Taran Killam','Ted Levine','Vince Gilligan','Warren Oates','Adam Arkin','Allan Hawco','Anna Camp','Bonnie Wright','Burt Young','Clayne Crawford','Cristine Rose','David Alpay','David Kelly','David Moscow','Dee Wallace','Demetri Martin','Devon Bostick','Dichen Lachman','Dyan Cannon','Elizabeth Gillies','Emmanuelle Vaugier','Fred Gwynne','Gail OGrady','Georgia King','Hugh Dillon','Ian Holm','Isabelle Carr','Jack Lemmon','Jeff Perry','Jenni Jwoww Farley','Jill Zarin','Jo Champa','Joel Schumacher','John Abraham','Jonathan Harris','Jonathan Silverman','Katie Leclerc','Katie Stevens','Kevin Daniels','Kiely Williams','Laura Leighton','LeVar Burton','Lee Thompson Young','Len Goodman','Liam Cunningham','Lou Ferrigno','Louis Garrel','Louis Gossett Jr.','Luke Macfarlane','Lynn Redgrave','Matt Lucas','Michael Kelly','Michael Shanks','Michael Stipe','Mike Nichols','Nona Gaye','Paul McGann','Pedro Pascal','Reece Thompson','Robbie Coltrane','Ryan Cartwright','Santiago Cabrera','Suzy Amis','Todd Phillips','Tom Riley','Tony Curran','William Devane','Xzibit','Abhishek Bachchan','Alan Thicke','Alden Ehrenreich','Alexis Denisof','Alona Tal','Amanda De Cadenet','Andrea Anders','Angela Lansbury','Anita Briem','Arthur Darvill','Bobby Farrelly','Brian Geraghty','Brian White','Bruno Kirby','Carroll OConnor','Chuck Wicks','Clmence Posy','Colm Feore','Daniel Auteuil','Daveigh Chase','David Keith','Debra Jo Rupp','Efren Ramirez','Eion Bailey','Elsa Pataky','Emma Kenney','Frank Miller','Freema Agyeman','George C. Scott','Glenda Jackson','Grace Phipps','Grard Jugnot','Hans Zimmer','Iain Glen','Jane Alexander','Jay R. Ferguson','Johnny Simmons','Justin Guarini','Karen Allen','Kayla Ewell','Kel Mitchell','Kelly McGillis','Keshia Knight Pulliam','Kieran Culkin','Kurt Fuller','Laura Innes','Liam Payne','Lyndie Greenwood','Mahershala Ali','Malcolm Barrett','Mariel Hemingway','Marla Sokoloff','Nelly','Nicolas Winding Refn','Olesya Rulin','Patty Duke','Paulina Gaitan','Peter Paige','Powers Boothe','Rachel Ticotin','Raymond J. Barry','Ryan Bingham','Sabrina Lloyd','Sebastian Koch','Sherilyn Fenn','Sophie Lowe','Sydney Pollack','Tammin Sursok','Tracey Ullman','Wes Craven','William Forsythe','Zo Bell','Aaron Sorkin','Alain Delon','Alyson Stoner','Amanda Kimmel','Annabelle Wallis','Annie Ilonzeh','April Bowlby','Ashley Rickards','Bill Engvall','Bojana Novakovic','Bonnie Bedelia','Bret McKenzie','Brittany Daniel','Bruno Tonioli','Buzz Aldrin','Carly Schroeder','Carol Burnett','Casey James','Charlie McDermott','Charlotte Ross','Clive Standen','D.C. Douglas','Dakota Blue Richards','Devon Aoki','Diane Neal','Drew Roy','Elena Anaya','Elisa Donovan','Elisabeth Rhm','Ethan Cutkosky','Francis Lawrence','Frank Whaley','Graham Greene','Hal Sparks','Hector Echavarria','Isabelle Fuhrman','Jake Weber','Jaleel White','James Tupper','Jemima Kirke','Jenifer Lewis','Jenny Slate','Jessica Walter','Jim Gaffigan','Joe Spano','Joe Swanberg','John Lloyd Young','John Waters','Kim Zolciak-Biermann','Kip Pardue','Kristen Hager','Lauren Lee Smith','Lee Van Cleef','Lily Cole','Maggie Siff','Matt Walsh','Michael Winterbottom','Mike The Situation Sorrentino','Nadine Velazquez','Niall Horan','Noah Baumbach','Olivier Martinez','Omar Sharif','Oscar Nuez','Pam Dawber','Paz de la Huerta','Peter Mullan','Salman Khan','Sarah Lancaster','Shawn Levy','Shirley Jones','Sophie McShera','Theo Rossi','Tim Matheson','Tony Chiu Wai Leung','Adam McKay','Addison Timlin','Aidan Gillen','Aidan Turner','Alexis Dziena','Armand Assante','Arsenio Hall','Bjrk','Brenda Blethyn','Brigid Brannagh','Callum Blue','Charles Grodin','Chelsea Handler','Christian Serratos','Dave Chappelle','David E. Kelley','Dirk Bogarde','Edward Asner','Ellen Muth','Enuka Okuma','Gregory Hines','Howard Stern','Iman','Jamel Debbouze','Jamie Dornan','Jane Kaczmarek','Jeff Foxworthy','Jesse James','Jessy Schram','Joe Wright','John Candy','John Savage','Josh Zuckerman','Katherine Helmond','Kelly Brook','Kim Richards','Lasse Hallstrm','Lauren Luna Vlez','Logan Huffman','Martin Donovan','Michael Bolton','Natalie Gal','Parker Young','Patrick Fischler','Peter Farrelly','Rachel Hurd-Wood','Rami Malek','Rex Harrison','Ricardo Montalban','Robert John Burke','Robert Loggia','Roberto Benigni','Shanola Hampton','Sherry Lansing','Stacy Keibler','Taylor Spreitler','Thomas Sadoski','Alex McCord','Allie Grant','Andrew Daly','Billy Boyd','Christine Lakin','Craig Bierko','Dana Plato','Declan Reynolds','Donnie Yen','Ed Begley Jr.','Ellen Wong','Eva Marie Saint','Frank Grillo','Fred Durst','Georgie Henley','Gregg Sulkin','Holly Robinson Peete','James Earl Jones','James Marsters','James Thomas Jr.','Jason Robards','Jill Scott','Jon Michael Davis','Jonathan Frid','Joseph Kosinski','Jrmie Renier','Justin Chon','Kara DioGuardi','Katy Mixon','Kaylee DeFer','Kellie Shanygne Williams','Kirk Douglas','Liv Ullmann','Louis Ferreira','LuAnn de Lesseps','Mackenzie Foy','Madison Pettis','Mario Lopez','Michael Biehn','Monet Mazur','Nigel Barker','Pamela Adlon','Pharrell Williams','Rick Yune','Riki Lindhome','Robert Forster','Ruth Wilson','Sally Pressman','Samantha Mathis','Seth MacFarlane','Sherri Shepherd','Spencer Grammer','Stephen Wallem','Tye Sheridan','Werner Herzog','Zoey Deutch','Anna Karina','Annabella Sciorra','Ayelet Zurer','Bob Newhart','Brooke Langton','Carey Lowell','Chris Carmack','Colin Egglesfield','Dong-gun Jang','Dot-Marie Jones','Eamonn Walker','Edward Zwick','Elizabeth Pea','Elle Macpherson','Emily Hampshire','Florence Henderson','Frances OConnor','French Stewart','Gregg Henry','James Kyson','Jean-Louis Trintignant','Jonathan Groff','Kathryn Bigelow','Keisha Castle-Hughes','Keith Simanton','Kelly Bensimon','Marcello Mastroianni','Maria Canals-Barrera','Marilu Henner','Marisol Nichols','Mary Tyler Moore','Michael Rady','Michael Stahl-David','Nicole Snooki Polizzi','Nicole Richie','Parvati Shallow','Paul Thomas Anderson','Quinton Aaron','Rachel Roberts','Richard Kind','Robert A. Iger','Rosie Huntington-Whiteley','Ryan McPartlin','Shannon Tweed','Ted McGinley','Todd Haynes','Traylor Howard','Victoria Beckham','Wim Wenders','Alfonso Ribeiro','Amy Schumer','Bennett Miller','Beth Grant','Brad Rowe','Chris Marquette','Conrad Bain','David Morse','DeForest Kelley','Diedrich Bader','Dove Cameron','Driton Tony Dovolani','Dwight Yoakam','Edi Gathegi','Estella Warren','Eva LaRue','Fionnula Flanagan','Girard Swan','Harold Ramis','Harry Lennix','Harry Styles','Hayes MacArthur','Ian McDiarmid','Ivan Reitman','James Coburn','Janet McTeer','Jay Harrington','Jonathan Demme','Julie Walters','Justin Bruening','Kandi Burruss','Karen Black','Katrina Bowden','Keely Shaye Smith','Kelly Ripa','Kym Whitley','Laura Benanti','Linda Hunt','Linus Roache','Lisa Vanderpump','Marc Warren','Marisa Coughlan','Matt Long','Melonie Diaz','Mike OMalley','Nate Torrence','Naturi Naughton','Nick Searcy','Oliver Reed','Rachel Hunter','Rob Riggle','Robert Llewellyn','Robert Shaw','Sam Heughan','Serinda Swan','Sonequa Martin-Green','T.I.','Walter Salles','Wei Tang','Will.i.am','Alex Kingston','Alexandra Breckenridge','Andrew Lee Potts','Ashton Holmes','Atom Egoyan','Beth Stern','Bianca Kajlich','Billy Gardell','Billy Unger','Blair Brown','Bruce Boxleitner','Charles Bronson','Charlotte Sullivan','Chris Jericho','Dana Carvey','Darius McCrary','Daryl Sabara','Davis Guggenheim','Douglas Booth','Drew Carey','Drew Seeley','Eugenio Derbez','Gaelan Connell','Gaius Charles','George Newbern','Gwendoline Christie','Harry Treadaway','Hrithik Roshan','Jay Leno','Jorma Taccone','Krista Allen','Lee DeWyze','Ne-Yo','Rachel Boston','Randy Couture','Richard Coyle','Ricky Schroder','Rod Steiger','Rodney Dangerfield','Ryan Murphy','Sarah Carter','Shakira','Shawn Hatosy','Shelley Conn','Sheryl Lee','Tahmoh Penikett','Terry Kinney','Tom Verica','Tony Bennett','Tony Scott','Willie Nelson','Xavier Samuel','Aimee Garcia','Alexander Gould','Amy Irving','Astrid Bergs-Frisbey','Bob Saget','Bridgit Mendler','Chad Johnson','Charlotte Le Bon','Cheyenne Jackson','Chris Zylka','Danny Gokey','David Bradley','David Gordon Green','Drake Bell','Elyse Levesque','Eric Idle','Franco Nero','Gia Mantegna','Hal Holbrook','Heather Matarazzo','Holly Madison','Ivana Milicevic','Jewel Kilcher','Joanna Cassidy','Johnny Weir','Julie Gonzalo','Kenny Johnson','Kim Dickens','Larry the Cable Guy','Marsha Thomason','Matt Passmore','Megalyn Echikunwoke','Naomi Campbell','Paige Turco','Paul Sorvino','Peter Boyle','Phil Hartman','Rachel Miner','Robert Taylor','Ruby Jerins','Stacey Dash','Tim Meadows','Vincenzo Amato','Will Poulter','Willow Smith','Alex Winter','Alexa Davalos','Amanda Schull','Andy Richter','Barry Levinson','Brad Bird','Brady Corbet','Brooke Burns','Callum Keith Rennie','Cassie Scerbo','Chris Vance','Danny Boyle','Danny Pino','Dylan Baker','Elise Neal','Emmanuelle Seigner','Ernest Borgnine','Fantasia Barrino','Gale Anne Hurd','Harry Belafonte','Helen McCrory','Henry Simmons','Hope Solo','Isaac Mizrahi','Isabella Vosmikova','James Le Gros','Jason Dohring','Jeremy Irvine','Joe Roth','Josh Lawson','Jurnee Smollett-Bell','Kate Gosselin','Katie Lohmann','Kerr Smith','Lambert Wilson','Laura Ramsey','Lauren Bacall','Laurence Olivier','Marilyn Manson','Michael Mosley','Michael Stuhlbarg','Michelle Borth','Miles Heizer','Miranda Cosgrove','Mischa Barton','Noah Bean','Oksana Baiul','Olivia Cooke','Peter Coyote','Peter MacNicol','Rob Kardashian','Rowan Blanchard','Rupert Everett','Scout Taylor-Compton','Sunny Mabrey','Tamsin Greig','Tempestt Bledsoe','Tracie Thoms','Will Sasso','Yannick Bisson','Aaron Tveit','Adam DeVine','Amanda Crew','Bernadette Peters','Camille Grammer','Charles Dance','Corey Haim','Dean Winters','Dominic Fumusa','Don Knotts','Dylan Minnette','Eddie Marsan','Ellar Coltrane','F. Murray Abraham','Holly Marie Combs','Jason Gedrick','Jennifer Ehle','Jes Macallan','Joel Courtney','John Mahoney','Kanye West','Kelli Williams','Kelly Blatz','Kevin Feige','Kevin Sussman','Lauren Ambrose','Liana Liberato','Liane Balaban','Lindsey Shaw','Lucas Neff','Macy Gray','Marton Csokas','Michael Steger','Mira Nair','Nicola Peltz','Nonso Anozie','Paul McCartney','Priyanka Chopra','Rory Cochrane','Samm Levine','Sienna Guillory','Sinbad','Taylor Hicks','Toby Regbo','Tyler Hoechlin','Xander Berkeley','Zack Snyder','Adam Rayner','Bear Grylls','Bebe Neuwirth','Bryan Brown','Candace Cameron Bure','Cassidy Freeman','Ccile De France','Charles Esten','Cheech Marin','Cliff Curtis','Colin Ford','D.B. Sweeney','Danny Masterson','David Letterman','Deepika Padukone','Eloise Mumford','Harry Shum Jr.','Illeana Douglas','Imelda Staunton','Jamie Bamber','Jane Adams','Jane Curtin','Jane Leeves','Jenna Boyd','John Larroquette','Jordan Peele','Judd Nelson','Julian Schnabel','Katherine Moennig','Kathy Baker','Kathy Najimy','Kiele Sanchez','Mark Moses','Matthew Weiner','Nathan Gamble','Nia Vardalos','Noah Taylor','Rick Fox','Shaun Sipos','Stephen Dillane','Steven Pasquale','Sugar Ray Leonard','Thomas Mann','Travis Milne','Vincent Perez','Will Patton','Yaya DaCosta','Aimee Teegarden','Alia Shawkat','Ana Gasteyer','Antonio Sabato Jr.','Armin Mueller-Stahl','Avan Jogia','Ben Savage','Booboo Stewart','Carl Reiner','Chaz Bono','Colton Haynes','Cress Williams','David Gere','David Paymer','Dennis Farina','Erin Cummings','Ernie Hudson','Guy Ritchie','Ice-T','Jaime Ray Newman','Jake Short','James Corden','James Lipton','Jaslene Gonzalez','Jo Marie Payton','John Simm','Joshua Gomez','Katie Aselton','Lauren Conrad','Lee Daniels','Lorraine Bracco','Madeleine Martin','Malcolm-Jamal Warner','Martn Lombard','Mdchen Amick','Mia Sara','Michael Trevino','Nat Wolff','Olivia Holt','Paulina Porizkova','Richard Attenborough','Sam Riley','Sarah Palin','Scott Wilson','Soleil Moon Frye','Steven Strait','Steven Weber','Talia Shire','Tyler Posey','Wendell Pierce','A.J. Cook','Amy Landecker','Andrea Riseborough','Annie Parisse','Beau Garrett','Blair Redford','Blake Jenner','Brenda Strong','Brian Benben','C.S. Lee','Carly Chaikin','Casper Van Dien','Celia Weston','Chris Lowell','Chris Parnell','Claire Coffee','Craig Ferguson','Crystal Bowersox','Cyndi Lauper','Deborah Kara Unger','Dylan OBrien','Frederick Weller','Geoffrey Arend','Gus Van Sant','Ian Harding','Jack Huston','Jason Beghe','Jason Priestley','Jim Jarmusch','Jimmy Bennett','Jon Turteltaub','Jonathan Jackson','Joshua Leonard','Karolina Kurkova','Kevin Corrigan','Kevin Durand','Lauren German','Lee Marvin','Len Wiseman','Lori Loughlin','Lucas Black','Malik Yoba','Max Burkholder','McKenzie Westmore','Michael Urie','Michel Piccoli','Nancy McKeon','Natalie Martinez','Neil Young','Paul Schulze','Rachel Zoe','Reid Scott','Rick Hoffman','Russell Hantz','Ryan Guzman','Sam Shepard','Scoot McNairy','Scott Michael Foster','Sergio Daz','Sonya Walger','Spike Jonze','Stephen Fry','Tom Everett Scott','Walter Matthau','Zosia Mamet','Anna Chlumsky','Bob Morley','Boris Kodjoe','Brandon T. Jackson','Brnice Bejo','Brnice Marlohe','Carl Weathers','Carmen Ejogo','Carrie Preston','Chris Barrie','Craig Sheffer','Crispin Glover','David Archuleta','Debbie Allen','Denis OHare','Dianne Wiest','Doris Roberts','Doug Jones','Graham McTavish','Gregory Peck','Harry Dean Stanton','Harry Shearer','JD Pardo','James Frain','Jessalyn Gilsig','Joey Lauren Adams','M. Night Shyamalan','Melissa Benoist','Melissa Etheridge','Michael Raymond-James','Monica Raymund','Paul Schneider','Reba McEntire','Robert Morse','Robert Zemeckis','Sean Maguire','Shah Rukh Khan','Tatum ONeal','Terence Stamp','The Edge','Adelaide Clemens','Ana Ortiz','Ann-Margret','Bailee Madison','Brent Spiner','Brian Baumgartner','Carice van Houten','Carrie Ann Inaba','Chris Diamantopoulos','Christine Woods','Claire Forlani','Constance Marie','Curtis Stone','Diablo Cody','Elizabeth Taylor','Emily Procter','Enver Gjokaj','Fanny Ardant','George Wendt','Keith Richards','Laura Marano','Manish Dayal','Marc Forster','Melora Hardin','Melora Walters','Michael Kors','Michael Weston','Rocky Carroll','Sam Jaeger','Samaire Armstrong','Shannen Doherty','Skylar Astin','Stefanie Scott','Stephen Mangan','Victor Webster','Vinessa Shaw','Akshay Kumar','Alan Ruck','Bethany Joy Lenz','Bonnie Somerville','Calum Worthy','David Hyde Pierce','Eddie Kaye Thomas','Eric Ladin','Fred Savage','Jamie Blackley','Jamie Campbell Bower','Jasmine Guy','Jay Duplass','Jonathan Tucker','Kadeem Hardison','Kenny Ortega','Lara Flynn Boyle','Mare Winningham','Michael Kenneth Williams','NeNe Leakes','Philip Baker Hall','RJ Mitte','Rachel Blanchard','Ramon Rodriguez','Rebecca Gayheart','Rock Hudson','Shantel VanSanten','T.R. Knight','Terry Gilliam','Thomas Ian Nicholas','Tia Mowry-Hardrict','Toni Braxton','Whitney Houston','Anne Bancroft','Antoine Fuqua','Austin Nichols','Barbara Walters','Barry Sloane','Beverly DAngelo','Brian Dennehy','C. Thomas Howell','Camilla Luddington','China Anne McClain','Courtney B. Vance','Courtney Thorne-Smith','Craig Charles','Dave Bautista','David Thewlis','Fisher Stevens','Franois Goeske','Gena Rowlands','Georgina Haig','Jackson Rathbone','Janet Montgomery','Jason Behr','John Lasseter','Jonathan Sadowski','Keegan-Michael Key','Ken Leung','Kristen Schaal','Leven Rambin','Mary Elizabeth Mastrantonio','Max Ryan','Meat Loaf','Mike Tyson','Nate Corddry','Neil Grayston','Nikki Blonsky','Nikolai Kinski','Penelope Ann Miller','Peter Stormare','RZA','Rob Brown','Robert Buckley','Sam Raimi','Steven Bauer','Steven Tyler','Aaron Carter','Aaron Yoo','Adewale Akinnuoye-Agbaje','Alanis Morissette','Anna Deavere Smith','Ben Bass','Chris Hardwick','Clare Bowen','Craig Anthony Olejnik','Damon Wayans','Dave Franco','Emma Dumont','Eve','Isaiah Washington','Italia Ricci','James Badge Dale','Jami Gertz','Joe Anderson','Lance Henriksen','Meghan Markle','Michael B. Jordan','Michael Mann','Michael York','Michelle Dockery','Mindy Cohn','Scott Adsit','Simon Phillips','Swoosie Kurtz','Telma Hopkins','Vanessa Ferlito','Wayne Brady','Zendaya']
	count = 0
	for row in range(rows):
		if row % 10000 == 0:
			print("Extracting... " + str(row) + "/" + str(rows) + '\r')
			
		if imdb[name_column][0][row][0] not in people:
			continue

		if str(imdb[faceprob_column][0][row]) == "-inf":
			continue
		try:
			save_image(input_dir + '/' + imdb[file_column][0][row][0],
						output_dir + '/' + imdb[name_column][0][row][0] + "." +  str(row) + ".jpg",
						imdb[coordinate_column][0][row][0])
			count += 1
		except:
			error = 1
			
	print("Extracting... " + str(rows) + "/" + str(rows))
	print("Extracted " + str(count) + " images!");
			
	return True
	
	
if __name__ == "__main__":
	generate_test_sets('/data/Imdb-Wiki/imdb/imdb.mat', '/data/Imdb-Wiki/imdb', '/data/Imdb-Wiki/imdb/testsets')
