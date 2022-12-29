drop database if exists predicted_outputs;
create database if not exists predicted_outputs;
use predicted_outputs;

drop table if exists predicted_outputs;
create table predicted_outputs(
transportation_expense int not null,
distance_to_work int not null,
daily_work_load_avg int not null,
body_mass_index int not null,
education bit not null,
children int not null,
pets int not null,
day_of_week int not null,
absence_reason_1 bit not null,
absence_reason_2 bit not null,
absence_reason_3 bit not null,
absence_reason_4 bit not null, 
probability float not null,
prediction bit not null);

select * from predicted_outputs
 
