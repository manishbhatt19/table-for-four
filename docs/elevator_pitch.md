# Table for Four: 90 Second Elevator Pitch

**CMU AI Agent Certification · Capstone**
**Author:** Manish Bhatt

> 231 words, which is about 90 seconds at a natural speaking pace. The four
> paragraphs are the problem, the system, the value, and the takeaway.

---

**The problem.** Booking dinner for four people sounds trivial until you do it. You
are juggling cuisine, area, party size, time, budget, somebody's gluten allergy, and
what is actually free on Friday. And it does not end in an answer, it ends in a
commitment: a table held, and four people expected somewhere at seven.

**The system.** I built Table for Four, an agentic concierge that takes one sentence,
"somewhere Italian for four on Friday", and carries it to a confirmed booking.
Behind it are four Model Context Protocol servers: restaurant search, a retrieval
layer over partner offers, a web lookup for what the food is like, and a booking
service with its own ledger. A LangGraph state machine runs the scripted path, and a
conversational host runs the chat.

**The value.** The part I care most about is who is allowed to do what. It is split
into five units, and the only one that runs a model holds no tools at all. It has to
ask. Every booking stops and waits for a person to press Reserve. And every reply is
checked against what the tools returned, so an invented time never reaches the
guest.

**The takeaway.** Every rule I wrote into the prompt eventually broke. Every rule I
moved into the code held. Agents are made trustworthy by what they cannot do, not by
what they are told.
